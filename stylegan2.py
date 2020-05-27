import tensorflow as tf

from stylegan2_generator import StyleGan2Generator
from stylegan2_discriminator import StyleGan2Discriminator

class StyleGan2(tf.keras.Model):
    """ 
    StyleGan2 config f for tensorflow 2.x 
    """
    def __init__(self, resolution=1024, weights=None, impl='cuda', gpu=True, **kwargs):
        """
        Parameters
        ----------
        resolution : int, optional
            Resolution output of the synthesis network, will be parsed
            to the floor integer power of 2. 
            The default is 1024.
        weights : string, optional
            weights name in weights dir to be loaded. The default is None.
        impl : str, optional
            Wether to run some convolutions in custom tensorflow
             operations or cuda operations. 'ref' and 'cuda' available.
            The default is 'cuda'.
        gpu : boolean, optional
            Wether to use gpu. The default is True.
        """
        super(StyleGan2, self).__init__(**kwargs)
        
        self.resolution = resolution
        if weights is not None:
            self.__adjust_resolution(weights)
        self.generator = StyleGan2Generator(resolution=self.resolution, weights=weights, 
                                            impl=impl, gpu=gpu, name='Generator')
        self.discriminator = StyleGan2Discriminator(resolution=self.resolution, weights=weights, 
                                                    impl=impl, gpu=gpu, name='Discriminator')
        
    def call(self, latent_vector):
        """
        Parameters
        ----------
        latent_vector : latent vector z of size [batch, 512].

        Returns
        -------
        score : output of the discriminator. 
        """
        img = self.generator(latent_vector)
        score = self.discriminator(img)

        return score    

    def __adjust_resolution(self, weights_name):
        """
        Adjust resolution of the synthesis network output. 
        
        Parameters
        ----------
        weights_name : name of the weights
        """
        if  weights_name == 'ffhq': 
            self.resolution = 1024
        elif weights_name == 'car': 
            self.resolution = 512
        elif weights_name in ['cat', 'church', 'horse']: 
            self.resolution = 256
      
