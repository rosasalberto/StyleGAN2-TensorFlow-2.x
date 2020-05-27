import tensorflow as tf

from utils.utils_stylegan2 import get_weight_initializer_runtime_coef
from dnnlib.ops.upfirdn_2d import upsample_conv_2d, conv_downsample_2d

class Conv2DLayer(tf.keras.layers.Layer):
    """
    StyleGan2 discriminator convolutional layer
    """
    def __init__(self, fmaps, kernel, up=False, down=False, 
                               demodulate=True, resample_kernel=None, gain=1, use_wscale=True, lrmul=1, 
                               impl='cuda', gpu=True, **kwargs):
                
        super(Conv2DLayer, self).__init__(**kwargs)
        
        self.fmaps = fmaps
        self.kernel = kernel
        
        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.gain = gain
        self.use_wscale = use_wscale
        self.lrmul = lrmul
        self.impl = impl
        self.gpu = gpu
        
    def build(self, input_shape):
        
        self.init_std_w, self.runtime_coef_w = get_weight_initializer_runtime_coef(shape=[self.kernel, self.kernel, input_shape[1], self.fmaps],
                                                                                   gain=self.gain, use_wscale=self.use_wscale, lrmul=self.lrmul)
        
        self.weight = self.add_weight(name='weight', shape=(self.kernel,self.kernel, input_shape[1], self.fmaps), 
                                      initializer=tf.random_normal_initializer(0, self.init_std_w), trainable=True)
    
    def call(self, x):
        
        # multiply weight per runtime_coef
        w = tf.math.multiply(self.weight, self.runtime_coef_w)
        
        # Convolution with optional up/downsampling.
        if self.up:
            if self.gpu:
                x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=self.resample_kernel, impl=self.impl)
            else:
                x = tf.transpose(x, [0, 2, 3, 1])
                x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NHWC', k=self.resample_kernel, impl=self.impl, gpu=False)
                x = tf.transpose(x, [0, 3, 1, 2]) 
        elif self.down:
            if self.gpu:
                x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=self.resample_kernel, impl=self.impl)
            else:
                x = tf.transpose(x, [0, 2, 3, 1])
                x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NHWC', k=self.resample_kernel, impl=self.impl, gpu=False)
                x = tf.transpose(x, [0, 3, 1, 2]) 
        else:
            if self.gpu:
                x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')
            else:
                x = tf.transpose(x, [0, 2, 3, 1])
                x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NHWC', strides=[1,1,1,1], padding='SAME')
                x = tf.transpose(x, [0, 3, 1, 2])  
        return x