import tensorflow as tf

from layers.modulated_conv_2d_layer import ModulatedConv2DLayer

class SynthesisMainLayer(tf.keras.layers.Layer):
    """
    StyleGan2 synthesis network main layer
    """
    def __init__(self, fmaps, up=False, impl='cuda', gpu=True,**kwargs):
        
        super(SynthesisMainLayer, self).__init__(**kwargs)
        
        self.fmaps = fmaps
        self.up = up
        self.impl = impl
        self.gpu = gpu
        
        self.resample_kernel = [1,3,3,1]
        self.kernel = 3
        
        if self.up:
            self.l_name = 'Conv0_up'
        else:
            self.l_name = 'Conv1'
        
    def build(self, input_shape):
        
        self.noise_strength = self.add_weight(name=self.l_name+'/noise_strength', shape=[], initializer=tf.initializers.zeros(), trainable=True)
        self.bias = self.add_weight(name=self.l_name+'/bias', shape=(self.fmaps,), initializer=tf.random_normal_initializer(0,1), trainable=True)
        
        self.mod_conv2d_layer = ModulatedConv2DLayer(fmaps=self.fmaps, kernel=self.kernel, 
                                                up=self.up, resample_kernel=self.resample_kernel,
                                                 impl=self.impl, gpu=self.gpu, name=self.l_name)
        
    def call(self, x, dlatent_vect):
                
        x = self.mod_conv2d_layer(x, dlatent_vect)
        
        #randomize noise
        noise = tf.random.normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        
        # adding noise to layer
        x += tf.math.multiply(noise, tf.cast(self.noise_strength, x.dtype))
        
        # adding bias and lrelu activation
        x += tf.reshape(self.bias, [-1 if i == 1 else 1 for i in range(x.shape.rank)])
        x = tf.math.multiply(tf.nn.leaky_relu(x, 0.2), tf.math.sqrt(2.))
        
        return x