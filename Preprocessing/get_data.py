import os
from datasets import load_dataset

iterable_dataset = load_dataset("huggan/wikiart", split="train")
save_dir = "../Data/"
os.makedirs(save_dir, exist_ok = True) #if dir not made make it else nothing
iterable_dataset.save_to_disk(save_dir)