import tensorflow as tf

from layers.conv_2d_layer import Conv2DLayer

class FromRgbLayer(tf.keras.layers.Layer):
    """
    StyleGan2 discriminator From RGB layer
    """
    def __init__(self, fmaps, impl='cuda', gpu=True, **kwargs):
        
        super(FromRgbLayer, self).__init__(**kwargs)
        
        self.fmaps = fmaps
        self.impl = impl
        self.gpu = gpu
        
    def build(self, input_shape):
        
        self.conv2d_rgb = Conv2DLayer(fmaps=self.fmaps, kernel=1, impl=self.impl, gpu=self.gpu, name='FromRGB')
        
        self.rgb_bias = self.add_weight(name='FromRGB/bias', shape=(self.fmaps,), 
                                        initializer=tf.random_normal_initializer(0,1), trainable=True)
        
    def call(self, x, y):
        
        t = self.conv2d_rgb(y)

        #add bias and lrelu activation
        t += tf.reshape(self.rgb_bias, [-1 if i == 1 else 1 for i in range(t.shape.rank)])
        t = tf.math.multiply(tf.nn.leaky_relu(t, 0.2), tf.math.sqrt(2.))
        
        return t if x is None else x + t