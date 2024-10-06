import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from datasets import Dataset as HFDataset

class ImgDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)  # reshaping for channels
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNN(nn.Module):
    def __init__(self, num_classes, img_channels=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)  # assuming image size of 64x64
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)  # flattening
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNClassifier:
    def __init__(self, train_ds: HFDataset, val_ds: HFDataset, test_ds: HFDataset, unique_styles, batch_size=32, epochs=10, learning_rate=0.001):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.unique_styles = unique_styles
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x_train, y_train = self.get_x_y(self.train_ds)
        x_val, y_val = self.get_x_y(self.val_ds)
        x_test, y_test = self.get_x_y(self.test_ds)

        self.train_loader = DataLoader(ImgDataset(x_train, y_train), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(ImgDataset(x_val, y_val), batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(ImgDataset(x_test, y_test), batch_size=self.batch_size, shuffle=False)

        self.model = CNN(num_classes=len(unique_styles)).to(self.device)

    def get_x_y(self, ds: HFDataset):
        return ds['img_pixels'], ds['label']

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            running_loss = 0.0
            self.model.train()

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {running_loss/len(self.train_loader):.4f}')

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=self.unique_styles))

    def run(self):
        self.train()
        self.evaluate()
