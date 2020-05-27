import tensorflow as tf

class MinibatchStdLayer(tf.keras.layers.Layer):
    """
    StyleGan2 discriminator MinibatchStdLayer
    """
    def __init__(self, group_size=4, num_new_features=1, **kwargs):
        
        super(MinibatchStdLayer, self).__init__(**kwargs)
        
        self.group_size = group_size
        self.num_new_features = num_new_features
        
    def call(self, x):
        
        group_size = tf.minimum(self.group_size, x.shape[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, self.num_new_features, s[1]//self.num_new_features, s[2], s[3]])   # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
        y = tf.cast(y, tf.float32)                              # [GMncHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMncHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MncHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MncHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[2,3,4], keepdims=True)      # [Mn111]  Take average over fmaps and pixels.
        y = tf.reduce_mean(y, axis=[2])                         # [Mn11] Split channels into c channel groups
        y = tf.cast(y, x.dtype)                                 # [Mn11]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [NnHW]  Replicate over group and pixels.
    
        return tf.concat([x, y], axis=1)