import tensorflow as tf
import numpy as np

class NonSaturatingLogLossPLRegularizer(tf.keras.losses.Loss):
    """ 
    Generator loss. Non-saturating logistic loss with path length regularizer from the paper
    Analyzing and Improving the Image Quality of StyleGAN", Karras et al. 2019
    """
    def __init__(self, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0, **kwargs):
        super(NonSaturatingLogLossPLRegularizer, self).__init__(**kwargs)
        self.pl_minibatch_shrink = pl_minibatch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean_var = 0
        
    def compute_logloss(self, G, D, minibatch_size):
        """
        Parameters
        ----------
        G : tensorflow model
            generator model.
        D : tensorflow model
            discriminator model.
        minibatch_size : int
            bathc_size.
        """
        latents = tf.random.normal([minibatch_size, 512])
        fake_images_out = G(latents)
        fake_scores_out = D(fake_images_out)
        loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
        return loss
    
    def compute_pathreg(self, G, D, minibatch_size):
        """
        Path length regularizer

        Parameters
        ----------
        G : tensorflow model
            generator model.
        D : tensorflow model
            discriminator model.
        minibatch_size : int
            bathc_size.
        """
        pl_latents = tf.random.normal([minibatch_size, 512])
        fake_dlatents_out = G.mapping_network(pl_latents)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(fake_dlatents_out)
            fake_images_out = G.synthesis_network(fake_dlatents_out)
            # Compute |J*y|.
            pl_noise = tf.random.normal(tf.shape(fake_images_out)) / np.sqrt(np.prod(fake_images_out.shape[2:]))
            lss = tf.reduce_sum(fake_images_out * pl_noise)
        pl_grads = tape.gradient(lss, fake_dlatents_out)
        pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))
        # Track exponential moving average of |J*y|.
        pl_mean = self.pl_mean_var + self.pl_decay * (tf.reduce_mean(pl_lengths) - self.pl_mean_var)
        self.pl_mean_var = pl_mean
        # Calculate (|J*y|-a)^2.
        pl_penalty = tf.square(pl_lengths - pl_mean)
        reg = pl_penalty * self.pl_weight
        return tf.reshape(reg,[-1,1])
    
class LogLossR1Regularizer(tf.keras.losses.Loss):
    """ 
    Discriminator Loss. R1 and R2 regularizers from the paper
    Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018
    """
    def __init__(self, gamma=10.0, **kwargs):
        super(LogLossR1Regularizer, self).__init__(**kwargs)
        self.gamma = gamma
        
    def compute_logloss(self, G, D, minibatch_size, reals):
        """
        Parameters
        ----------
        G : tensorflow model
            generator model.
        D : tensorflow model
            discriminator model.
        minibatch_size : int
            bathc_size.
        reals : tensor
        real images tensor.
        """
        latents = tf.random.normal([minibatch_size, 512])
        fake_images_out = G(latents)
        fake_scores_out = D(fake_images_out)
        real_scores_out = D(reals)
        loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
        loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) 
        return loss
    
    def compute_r1reg(self, G, D, minibatch_size, reals):
        """
        Parameters
        ----------
        G : tensorflow model
            generator model.
        D : tensorflow model
            discriminator model.
        minibatch_size : int
            bathc_size.
        reals : tensor
        real images tensor.
        """
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(reals)
            real_scores_out = D(reals)
            real_grads = tape.gradient(tf.reduce_sum(real_scores_out), reals)
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        reg = gradient_penalty * (self.gamma * 0.5)
        return tf.reshape(reg,[-1,1])
