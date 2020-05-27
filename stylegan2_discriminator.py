import tensorflow as tf
import numpy as np

from utils.weights_map import available_weights, weights_stylegan2_dir, discriminator_weights
from utils.utils_stylegan2 import nf
from layers.mini_batch_std_layer import MinibatchStdLayer
from layers.from_rgb_layer import FromRgbLayer
from layers.block_layer import BlockLayer
from layers.conv_2d_layer import Conv2DLayer
from layers.dense_layer import DenseLayer

class StyleGan2Discriminator(tf.keras.layers.Layer):
    """
    StyleGan2 discriminator config f for tensorflow 2.x 
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
        super(StyleGan2Discriminator, self).__init__(**kwargs)
    
        self.gpu = gpu
        self.impl = impl
    
        self.resolution = resolution
        if weights is not None: self.__adjust_resolution(weights)
        self.resolution_log2 = int(np.log2(resolution))
        
        # load weights
        if weights is not None:
            _ = self(tf.zeros(shape=(1, 3, self.resolution, self.resolution)))
            self.__load_weights(weights)
             
    def build(self, input_shape):
        
        self.mini_btch_std_layer = MinibatchStdLayer()
        self.from_rgb = FromRgbLayer(fmaps=nf(self.resolution_log2-1), 
                                     name='{}x{}'.format(self.resolution, self.resolution),
                                     impl=self.impl, gpu=self.gpu)
        
        for res in range(self.resolution_log2, 2, -1):
            res_str = str(2**res)
            setattr(self, 'block_{}_{}'.format(res_str, res_str), 
                    BlockLayer(res=res, name='{}x{}'.format(res_str, res_str), 
                               impl=self.impl, gpu=self.gpu))
        
        #last layers
        self.conv_4_4 = Conv2DLayer(fmaps=nf(1), kernel=3, impl=self.impl, 
                                    gpu=self.gpu, name='4x4/Conv')
        self.conv_4_4_bias = self.add_weight(name='4x4/Conv/bias', shape=(512,),
                                             initializer=tf.random_normal_initializer(0,1), trainable=True)
        self.dense_4_4 = DenseLayer(fmaps=512, name='4x4/Dense0')
        self.dense_output = DenseLayer(fmaps=1, name='Output')
    
    def call(self, y):
        """

        Parameters
        ----------
        y : tensor of the image/s to evaluate. shape [batch, channel, height, width]

        Returns
        -------
        output of the discriminator. 

        """
        y = tf.cast(y, 'float32')
        x = None
        
        for res in range(self.resolution_log2, 2, -1):
            if  res == self.resolution_log2:
                x = self.from_rgb(x, y)
            x = getattr(self, 'block_{}_{}'.format(2**res, 2**res))(x)

        #minibatch std dev
        x = self.mini_btch_std_layer(x)
        
        #last convolution layer
        x = self.conv_4_4(x)
        x += tf.reshape(self.conv_4_4_bias, [-1 if i == 1 else 1 for i in range(x.shape.rank)])
        x = tf.math.multiply(tf.nn.leaky_relu(x, 0.2), tf.math.sqrt(2.))
        
        x = tf.reshape(x, [-1, np.prod([d for d in x.shape[1:]])])
        # dense layer
        x = self.dense_4_4(x)
        x = tf.math.multiply(tf.nn.leaky_relu(x, 0.2), tf.math.sqrt(2.))
        #output layer
        x = self.dense_output(x)
        
        return tf.identity(x, name='scores_out')
    
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
    
    def __load_weights(self, weights_name):
        """
        Load pretrained weights, stored as a dict with numpy arrays.
        Parameters
        ----------
        weights_name : name of the weights

        """
        
        if (weights_name in available_weights) and type(weights_name) == str:
            data = np.load(weights_stylegan2_dir + weights_name + '.npy', allow_pickle=True)[()]
            
            weights_discriminator = [data.get(key) for key in discriminator_weights[weights_name]]
            self.set_weights(weights_discriminator)
            
            print("Loaded {} discriminator weights!".format(weights_name))
        else:
            print('Cannot load the specified weights')