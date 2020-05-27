import tensorflow as tf

from utils.utils_stylegan2 import get_weight_initializer_runtime_coef

class DenseLayer(tf.keras.layers.Layer):
    """
    StyleGan2 Dense layer, including weights multiplication per runtime coef, and bias multiplication per lrmul
    """
    def __init__(self, fmaps, lrmul=1, **kwargs):
        
        super(DenseLayer, self).__init__(**kwargs)
        
        self.fmaps = fmaps
        self.lrmul = lrmul
        
    def build(self, input_shape):
        
        init_std, self.runtime_coef = get_weight_initializer_runtime_coef(shape=[input_shape[1], self.fmaps], 
                                                                              gain=1, use_wscale=True, lrmul=self.lrmul)
        
        self.dense_weight = self.add_weight(name='weight', shape=(input_shape[1],self.fmaps),
                                            initializer=tf.random_normal_initializer(0,init_std), trainable=True)
        self.dense_bias = self.add_weight(name='bias', shape=(self.fmaps,),
                                          initializer=tf.random_normal_initializer(0,init_std), trainable=True)
        
    def call(self, x):
        
        x = tf.matmul(x, tf.math.multiply(self.dense_weight, self.runtime_coef)) 
        x += tf.reshape(tf.math.multiply(self.dense_bias, self.lrmul), [-1 if i == 1 else 1 for i in range(x.shape.rank)])
        
        return x