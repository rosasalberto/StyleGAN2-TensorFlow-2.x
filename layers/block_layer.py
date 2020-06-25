import tensorflow as tf

from utils.utils_stylegan2 import nf
from layers.conv_2d_layer import Conv2DLayer

class BlockLayer(tf.keras.layers.Layer):
    """
    StyleGan2 discriminator Block layer
    """
    def __init__(self, res, impl='cuda', gpu=False, **kwargs):
        
        super(BlockLayer, self).__init__(**kwargs)
        
        self.res = res
        self.impl = impl
        self.gpu = gpu
        self.resample_kernel = [1, 3, 3, 1]
        
    def build(self, input_shape):
        
        self.conv2d_0 = Conv2DLayer(fmaps=nf(self.res-1), kernel=3, 
                                    impl=self.impl, gpu=self.gpu, name='Conv0')
        
        self.bias_0 = self.add_weight(name='Conv0/bias', shape=(nf(self.res-1),), 
                                      initializer=tf.random_normal_initializer(0, 1), trainable=True)
        
        self.conv2d_1_down = Conv2DLayer(fmaps=nf(self.res-2), kernel=3, down=True, 
                                         resample_kernel=self.resample_kernel, 
                                         impl=self.impl, gpu=self.gpu, name='Conv1_down')
        
        self.bias_1_down = self.add_weight(name='Conv1_down/bias', shape=(nf(self.res-2),), 
                                           initializer=tf.random_normal_initializer(0, 1), trainable=True)
        
        self.conv2d_skip = Conv2DLayer(fmaps=nf(self.res-2), kernel=1, down=True, 
                                       resample_kernel=self.resample_kernel, 
                                       impl=self.impl, gpu=self.gpu, name='Skip')
        
    def call(self, x):
        t = x
        
        x = self.conv2d_0(x)
        x += tf.reshape(self.bias_0, [-1 if i == 1 else 1 for i in range(x.shape.rank)])
        x = tf.math.multiply(tf.nn.leaky_relu(x, 0.2), tf.math.sqrt(2.))
        
        x = self.conv2d_1_down(x)
        x += tf.reshape(self.bias_1_down, [-1 if i == 1 else 1 for i in range(x.shape.rank)])
        x = tf.math.multiply(tf.nn.leaky_relu(x, 0.2), tf.math.sqrt(2.))
 
        t = self.conv2d_skip(t)
        x = (x + t) * (1 / tf.math.sqrt(2.))
        
        return x