import tensorflow as tf
import numpy as np

def get_weight_initializer_runtime_coef(shape, gain=1, use_wscale=True, lrmul=1):
    """ get initializer and lr coef for different weights shapes"""
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul
    
    return init_std, runtime_coef

def convert_images_to_uint8(images, drange=[-1, 1], nchw_to_nhwc=False, shrink=1, uint8_cast=True):
    """Convert a minibatch of images from float32 to uint8 with configurable dynamic range.
    Can be used as an output transformation for Network.run().
    """
    images = tf.cast(images, tf.float32)
    if shrink > 1:
        ksize = [1, 1, shrink, shrink]
        images = tf.nn.avg_pool(images, ksize=ksize, strides=ksize, 
                                padding="VALID", data_format="NCHW")
    if nchw_to_nhwc:
        images = tf.transpose(images, [0, 2, 3, 1])
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)
    if uint8_cast:
        images = tf.saturate_cast(images, tf.uint8)
    return images

def nf(stage, fmap_base=16 << 10, fmap_decay=1.0, fmap_min=1, fmap_max=512): 
    return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    