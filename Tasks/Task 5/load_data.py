from datasets import load_from_disk 

SAVE_DIR = '../processed_data'

# return x (i.e image pixel values) and y (encoded label (i.e list or label value))
def load_train_data(for_CNN=False, num_samples=None):
    if for_CNN:
        pass # return data of shape (batch_size,64,64,3)
    else: # return data of shape (n_rows, 64*64*3)
        ds = load_from_disk(SAVE_DIR)
        if num_samples != None:
            ds = ds.select(range(num_samples))
        return ds['img_pixels'], ds['y']
