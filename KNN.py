import os
import numpy as np
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch
import torchvision.transforms as transforms

iterable_dataset = load_dataset("huggan/wikiart", split="train", streaming=True)

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to a smaller size
    transforms.ToTensor(),        # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize image
])

# Initialize lists to store features and labels
images = []
artists = []
genres = []
styles = []

# Iterate over the dataset to collect data
for example in iterable_dataset:
    try:
        # Load and preprocess the image
        img = Image.open(example['image'])
        img = transform(img).numpy().flatten()  # Flatten the image for KNN
        images.append(img)

        # Collect other features
        artists.append(example['artist'])
        genres.append(example['genre'])
        styles.append(example['style'])
    except Exception as e:
        print(f"Error processing image: {e}")
        continue

# Convert lists to arrays
X = np.array(images)

# Encode categorical labels (artist, genre, style)
genre_encoder = LabelEncoder()
y = genre_encoder.fit_transform(genres)

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Fit KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"KNN Accuracy: {accuracy:.4f}")
