import tensorflow as tf

class DatasetLoader:
    """ Helper for load the dataset using tf.Dataset class """
    def __init__(self, path_dir, resolution, batch_size, cache_file=True):
        """
        path_dir : Directory of the image dataset
        resolution : Resolution of the training images
        batch_size : Batch size
        cache_file : filepath to store cache files .tfcache extension
        """
        self.resolution = resolution
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        
        list_ds = tf.data.Dataset.list_files(str(path_dir+'/*'))
        labeled_ds = list_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE)
        self.train_ds = self.prepare_for_training(labeled_ds, cache=self.cache_file)

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [self.resolution, self.resolution])

    def process_path(self, file_path):
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        return self.decode_img(img)
    
    def get_batch(self):
        images = next(iter(self.train_ds))
        return tf.transpose(images, [0, 3, 1, 2]) 

    def prepare_for_training(self, ds, cache=True, shuffle_buffer_size=100):
      # use `.cache(filename)` to cache preprocessing work for datasets that don't
      # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        # Repeat forever
        ds = ds.repeat()
        ds = ds.batch(self.batch_size)
        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds