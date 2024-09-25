import os
import pandas as pd
from datasets import load_from_disk


def load_and_batch_data(data_path, batch_size=1000):

    dataset = load_from_disk(data_path)

    # List to store each batched dataframe
    batches = []

    # Function to process each batch and convert it into pandas DataFrame
    def process_batch(batch):
        df_batch = pd.DataFrame(batch)
        batches.append(df_batch)
        return batch  # Required by the map function

    # Apply the map function to process the dataset in batches
    dataset.map(process_batch, batched=True, batch_size=batch_size)

    return batches
