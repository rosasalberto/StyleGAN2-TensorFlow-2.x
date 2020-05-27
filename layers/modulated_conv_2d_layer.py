import tensorflow as tf
import numpy as np

from utils.utils_stylegan2 import get_weight_initializer_runtime_coef
from dnnlib.ops.upfirdn_2d import upsample_conv_2d, conv_downsample_2d

class ModulatedConv2DLayer(tf.keras.layers.Layer):
    """
    StyleGan2 generator modulated convolution layer
    """
    def __init__(self, fmaps, kernel, up=False, down=False, 
                               demodulate=True, resample_kernel=None, gain=1, use_wscale=True, lrmul=1, 
                               fused_modconv=True, impl='cuda', gpu=True, **kwargs):
                
        super(ModulatedConv2DLayer, self).__init__(**kwargs)
        
        self.fmaps = fmaps
        self.kernel = kernel
        
        self.up = up
        self.down = down
        self.demodulate = demodulate
        self.resample_kernel = resample_kernel
        self.gain = gain
        self.use_wscale = use_wscale
        self.lrmul = lrmul
        self.fused_modconv = fused_modconv
        self.latent_size = 512
        self.impl = impl
        self.gpu = gpu
        
    def build(self, input_shape):
        
        self.init_std_w, self.runtime_coef_w = get_weight_initializer_runtime_coef(shape=[self.kernel, self.kernel, input_shape[1], self.fmaps],
                                                                                   gain=self.gain, use_wscale=self.use_wscale, lrmul=self.lrmul)
        
        self.init_std_s, self.runtime_coef_s = get_weight_initializer_runtime_coef(shape=[self.latent_size, input_shape[1]],
                                                                                   gain=self.gain, use_wscale=self.use_wscale, lrmul=self.lrmul)
        
        self.mod_bias = self.add_weight(name='mod_bias', shape=(input_shape[1],), 
                                        initializer=tf.random_normal_initializer(0, self.init_std_s), trainable=True)
        self.mod_weight = self.add_weight(name='mod_weight', shape=(self.latent_size, input_shape[1]), 
                                          initializer=tf.random_normal_initializer(0, self.init_std_s), trainable=True)
        self.weight = self.add_weight(name='weight', shape=(self.kernel,self.kernel, input_shape[1], self.fmaps), 
                                      initializer=tf.random_normal_initializer(0, self.init_std_w), trainable=True)
    
    def call(self, x, dlatent_vect):
        
        # multiply weight per runtime_coef
        w = tf.math.multiply(self.weight, self.runtime_coef_w)
        ww = w[np.newaxis] # [BkkIO] Introduce minibatch dimension.
        
        # Modulate.
        s = tf.matmul(dlatent_vect, tf.math.multiply(self.mod_weight, self.runtime_coef_s)) # [BI] Transform incoming W to style.
        s = tf.nn.bias_add(s, tf.math.multiply(self.mod_bias, self.lrmul)) + 1 # [BI] Add bias (initially 1).

        ww = tf.math.multiply(ww, tf.cast(s[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype)) # [BkkIO] Scale input feature maps.
        
        # Demodulate.
        if self.demodulate:
            d = tf.math.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3]) + 1e-8) # [BO] Scaling factor.
            ww = tf.math.multiply(ww, d[:, np.newaxis, np.newaxis, np.newaxis, :]) # [BkkIO] Scale output feature maps.
        
        # Reshape/scale input.
        if self.fused_modconv:
            x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]]) # Fused => reshape minibatch to convolution groups.
            w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww.shape[1], ww.shape[2], ww.shape[3], -1])
        else:
            x = tf.math.multiply(x, tf.cast(s[:, :, np.newaxis, np.newaxis], x.dtype)) # [BIhw] Not fused => scale input activations.
        
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
                  
        # Reshape/scale output.
        if self.fused_modconv:
            x = tf.reshape(x, [-1, self.fmaps, x.shape[2], x.shape[3]]) # Fused => reshape convolution groups back to minibatch.
        elif self.demodulate:
            x = tf.math.multiply(x, tf.cast(d[:, :, np.newaxis, np.newaxis], x.dtype)) # [BOhw] Not fused => scale output activations.
        return x