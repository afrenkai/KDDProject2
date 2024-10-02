from datasets import load_from_disk 
import tensorflow as tf
import numpy as np
import math
SAVE_DIR = '../processed_data/'
HEIGHT = 64
WIDTH = 64

# return x (i.e image pixel values) and label (encoded label (i.e list or label value))
def load_train_val_data(for_CNN=False, num_samples=None, val_size=0.2):
    path = SAVE_DIR
    if for_CNN:
        path += 'cnn' # shape = (HEIGHT, WIDTH)
    else: 
        path += 'regular' # shape = (HEIGHT*WIDTH)
    ds = load_from_disk(SAVE_DIR) # loads preprocessed data
    unique_styles = ds.unique('styles')
    unique_styles = np.array(unique_styles)

    
    if num_samples is not None:
        ds = ds.select(range(num_samples))
    
    ds = ds.train_test_split(test_size=val_size, shuffle=True)
    return ds['train'], ds['test'],  unique_styles

def get_tf_train_val_dataset(for_CNN=False, num_samples=None, val_size=0.2, batch_size=32):
    train_ds, val_ds, unique_styles = load_train_val_data()
    if for_CNN:
        shape = (HEIGHT, WIDTH)
        output_shape = (len(unique_styles))
    else:
        shape = (HEIGHT*WIDTH)
        output_shape = ()
    train_dataset = to_tf_dataset(train_ds, shape, output_shape, batch_size=batch_size)
    val_dataset = to_tf_dataset(val_ds, shape, output_shape, batch_size=batch_size)
    return train_dataset, val_dataset



# this should work differently for cnn vs other models
# other models would directly use the encoded labels (i.e "string" -> num)
# cnn needs (0,1,0,0) i.e. one hot encoded output for this
def to_tf_dataset(ds, shape, output_shape, batch_size=32):
    def generator():
        for row in ds:
            yield row['img_pixels'], row['label'] 
    dataset = tf.data.Dataset.from_generator(
        generator, 
        output_signature=(
            # tf.TensorSpec(shape=(HEIGHT, WIDTH, CHANNELS), dtype=tf.float32),  # Image shape
            tf.TensorSpec(shape=(shape), dtype=tf.float64),  # Image shape
            tf.TensorSpec(shape=(output_shape), dtype=tf.int64)  # Label shape, (nclass,) for cnn; () for others
        )
    )
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
    return dataset