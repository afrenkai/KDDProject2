from datasets import load_from_disk 

SAVE_DIR = '../processed_data/'

# return x (i.e image pixel values) and y (encoded label (i.e list or label value))
def load_train_data(for_CNN=False, num_samples=None):
    if for_CNN:
        path += 'cnn' # shape = (HEIGHT, WIDTH)
    else: 
        path += 'regular' # shape = (HEIGHT*WIDTH)
    ds = load_from_disk(SAVE_DIR)
    if num_samples != None:
        ds = ds.select(range(num_samples))
    return ds['img_pixels'], ds['y']
