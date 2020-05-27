import tensorflow as tf

from layers.modulated_conv_2d_layer import ModulatedConv2DLayer

class ToRgbLayer(tf.keras.layers.Layer):
    """
    StyleGan2 generator To RGB layer
    """
    def __init__(self, impl='cuda', gpu=True,**kwargs):
        
        super(ToRgbLayer, self).__init__(**kwargs)
        
        self.impl = impl
        self.gpu = gpu
        
    def build(self, input_shape):
        
        self.mod_conv2d_rgb = ModulatedConv2DLayer(fmaps=3, kernel=1, demodulate=False, 
                                              impl=self.impl, gpu=self.gpu, name='ToRGB')
        
        self.rgb_bias = self.add_weight(name='ToRGB/bias', shape=(3,), 
                                        initializer=tf.random_normal_initializer(0, 1), trainable=True)
        
    def call(self, x, dlatent_vect, y):
        
        u = self.mod_conv2d_rgb(x, dlatent_vect)
        t = u + tf.reshape(self.rgb_bias, [-1 if i == 1 else 1 for i in range(x.shape.rank)])
        
        return t if y is None else y + t