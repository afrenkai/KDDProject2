from datasets import load_dataset

# Load a dataset from Hugging Face
dataset = load_dataset('huggan/wikiart')

# Convert to pandas DataFrame
df = dataset['train'].to_pandas()  # Or 'validation', 'test', etc.

# Display the first few rows of the DataFrame
print(df.head())
