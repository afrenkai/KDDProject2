=Task 4: Preprocessing

Perform data processing on the data. Dive deep into data preparation, ensuring
that the dataset is primed and ready for model training. Document all pre-
processing steps and include them in the report, specify what happened to the
data after the step, why you performed it and if you are going to use it moving
forward. For example:
1. Data Cleaning: Address any inconsistencies within the data. This includes
dealing with missing values, anomalies, and duplicated entries.
2. Data Transformation and Normalization: Depending on the nature and
distribution of your data, apply necessary transformations. Ensure features
are on a similar scale, making them more amenable to analysis and model
training.
3. Specific Data Type Processing:
(a) Text Data: Implement tokenization to break down text into smaller
chunks, remove stop words to filter out common but uninformative
words, and utilize feature extraction techniques, like TF-IDF or word
embeddings, to represent text in a form suitable for machine learning.
(b) Image Data: Normalize image pixel values, ensuring they fall within
a consistent range (typically 0 to 1). Consider image augmenta-
tion techniques to artificially enhance your dataset, creating varied
representations of the same image to improve model robustness.