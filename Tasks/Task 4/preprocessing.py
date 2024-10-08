import tensorflow as tf
import numpy as np
from datasets import load_dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import perf_counter

# Constants
HEIGHT = 64
WIDTH = 64
CHANNELS = 3 # Should not matter if b&w
SAVE_PATH = '../processed_data/'
HF_MAP_BATCH_SIZE = 1000 # default = 1000 < 100 if u running on cuda
NUM_PROCS = 8
# FLAGS
CONVERT_TO_BW_FLAG = True
AUGMENT_IMAGE_FLAG = False

def get_unique_styles(ds: DatasetDict):
    return ds.unique('style')

def style2idx(style, style_list):
    return style_list.index(style)

def augment_with_tf(image, shape):
    # resize image
    image = image.resize((HEIGHT,WIDTH))
    # convert rgb image to grey scale
    if CONVERT_TO_BW_FLAG:
        image = image.convert('L') # better support for converting compared to tf.Image.rbg_to_grayscale
    if AUGMENT_IMAGE_FLAG:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # reshape img array to the desired shape
    image = tf.reshape(image, shape)
    image = np.array(image, dtype=np.float64)
    image = image * (1./255)
    return image

# augment over batches
def augment(examples, shape):
    examples['img_pixels'] = [augment_with_tf(img,shape) for img in examples['image']]
    return examples

# to be used in batches
def encode_labels(x, label_encoder: LabelEncoder, for_CNN):
    x['label'] = label_encoder.transform(np.array(x['style']).reshape(-1,1))
    if for_CNN:
        x['label']  = x['label'].toarray()
    return x

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


if __name__ == "__main__":
    start_time = perf_counter()
    print("Starting Dataset Preprocessing")
    print("Preprocessed dataset will be saved in '../processed_data/'")
    ds = load_dataset("jlbaker361/wikiart")

    # # limit for testing
    # ds['train'] = ds['train'].select(range(1000))
    # ds['test'] = ds['test'].select(range(1000))

    unique_styles_train = get_unique_styles(ds['train'])
    unique_styles_test = get_unique_styles(ds['test'])
    unique_styles: list = unique_styles_train + [x for x in unique_styles_test if x not in unique_styles_train]
    unique_styles = np.array(sorted(unique_styles))

    # preprocess train and test datasets
    for ds_type in ['train', 'test']: # for both train and test set
        ds_to_process = ds[ds_type]
        for for_CNN in [True, False]: # for CNN and other models
            print(f'Processing {ds_type} dataset, for CNN = {for_CNN}')
            if for_CNN:
                encoder = OneHotEncoder().fit(unique_styles.reshape(-1,1))
                shape = (HEIGHT, WIDTH, CHANNELS)
                output_shape = (len(unique_styles))
                if CONVERT_TO_BW_FLAG:
                    shape = (HEIGHT, WIDTH)
            else:
                encoder = LabelEncoder().fit(unique_styles)
                shape = (HEIGHT*WIDTH*CHANNELS)
                output_shape = ()
                if CONVERT_TO_BW_FLAG:
                    shape = (HEIGHT*WIDTH)

            ds_encoded = ds_to_process.map(lambda x: encode_labels(x, encoder, for_CNN), batched=True, batch_size=HF_MAP_BATCH_SIZE,
                                           num_proc=NUM_PROCS)
            ds_augmented = ds_encoded.map(lambda x: augment(x, shape), batched=True, batch_size=HF_MAP_BATCH_SIZE,
                                           num_proc=NUM_PROCS)
            path = get_save_dir(ds_type,for_CNN)
            ds_augmented.save_to_disk(path)
    print(f'Preprocessing Complete in {round(perf_counter()-start_time, 2)}s')

