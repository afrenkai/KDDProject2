import os
from datasets import load_dataset

iterable_dataset = load_dataset("huggan/wikiart", split="train", streaming=True)