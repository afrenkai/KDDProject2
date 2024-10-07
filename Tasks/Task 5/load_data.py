from datasets import load_from_disk, Dataset
import tensorflow as tf
import numpy as np
import math
from torch.utils.data import DataLoader

SAVE_PATH = '../processed_data/'
HEIGHT = 64
WIDTH = 64

def get_datasets(for_CNN=False, num_samples=None, val_size=0.2, batch_size=32):
    '''
    returns train_dataset [Type: tf.Dataset (batched) if batch_size != None], val_dataset, test_dataset , unique_styles
    '''
    train_ds, val_ds, test_ds, unique_styles = load_data(for_CNN=for_CNN, num_samples=num_samples, val_size=val_size)

    if batch_size == None:
        # return test dataset as is
        return train_ds, val_ds, test_ds, unique_styles
        
    # train_dataset = to_tf_dataset(train_ds, shape, output_shape, batch_size=batch_size)
    train_dataset = to_pytorch_dataloader(train_ds, batch_size=batch_size)
    test_dataset = test_ds

    if for_CNN:
        if val_ds != None:
            val_ds.set_format(type='torch', columns=['img_pixels', 'label'])
        test_dataset = to_pytorch_dataloader(test_ds, batch_size=batch_size)

    return train_dataset, val_ds, test_dataset, unique_styles

# returns save dir
def get_save_dir(ds_type: str, for_CNN: bool):
    path = SAVE_PATH
    path+= f'{ds_type}/'
    # save dataset
    if for_CNN:
        path += 'cnn' # input_shape = (HEIGHT, WIDTH), out_shape = (n_classes)
    else: 
        path += 'regular' # shape = (HEIGHT*WIDTH), out_shape = ()
    return path


# return x (i.e image pixel values) and label (encoded label (i.e list or label value))
def load_data(for_CNN=False, num_samples=None, val_size=0.2):
    '''
    returns train, val, test, unique_styles
    '''
    path_train = get_save_dir('train',for_CNN)
    path_test = get_save_dir('test', for_CNN)
    ds_train = load_from_disk(path_train) # loads preprocessed train data
    ds_test = load_from_disk(path_test) # loads preprocessed test data
    unique_styles_train = ds_train.unique('style')
    unique_styles_test = ds_test.unique('style') 
    unique_styles = unique_styles_train + [x for x in unique_styles_test if x not in unique_styles_train]
    unique_styles = np.array(sorted(unique_styles))

    if num_samples is not None:
        ds_train = ds_train.select(range(num_samples))
    
    if val_size <= 0.0:
        return ds_train, None,  ds_test, unique_styles
    else:
        ds_train = ds_train.train_test_split(test_size=val_size, shuffle=True)
        # train, val, test, unique_styles
        return ds_train['train'], ds_train['test'],  ds_test, unique_styles


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
    dataset = dataset.shuffle(buffer_size=1000)
    if batch_size != None:
        dataset = dataset.batch(batch_size)
    return dataset


def to_pytorch_dataloader(ds: Dataset, batch_size=32):
    ds.set_format(type='torch', columns=['img_pixels', 'label'])
    return DataLoader(ds, batch_size=batch_size, shuffle=True)