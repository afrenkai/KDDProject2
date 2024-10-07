import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset as HFDataset

class CNN(nn.Module):
    def __init__(self, num_classes, img_channels=1, dropout_rate=0.0):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout layer added :)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)  # assuming image size of 64x64
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)  # flattening 
        x = torch.relu(self.fc1(x))
        x = self.dropout(x) 
        x = self.fc2(x)
        return x


class CNNClassifier:
    def __init__(self, train_ds, val_ds, test_ds, unique_styles, batch_size=32, epochs=5, learning_rate=0.001, dropout_rate=0.0):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.unique_styles = unique_styles
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNN(num_classes=len(unique_styles), dropout_rate=dropout_rate).to(self.device)  # Include dropout_rate

    def get_x_y(self, ds):
        return ds['img_pixels'], ds['label']

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            running_loss = 0.0
            self.model.train(True)

            for _, data in enumerate(self.train_ds):
                images, labels = self.get_x_y(data)
                images, labels = images.to(self.device), labels.to(self.device)
                images = images.reshape(-1, 1, 64, 64)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            val_loss = 0.0
            with torch.no_grad():
                images, labels = self.get_x_y(self.val_ds)
                images, labels = images.to(self.device), labels.to(self.device)
                images = images.reshape(-1, 1, 64, 64)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {running_loss/len(self.train_ds):.4f}, Validation Loss: {val_loss/len(self.val_ds):.4f}')

    def evaluate(self,print_res=True):
        self.model.eval()
        all_preds = []
        all_labels = []
        running_val_loss = 0.0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data in self.test_ds:
                images, labels = self.get_x_y(data)
                images, labels = images.to(self.device), labels.to(self.device)
                images = images.reshape(-1, 1, 64, 64)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, actual = torch.max(labels, 1)

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(actual.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        val_loss = running_val_loss / len(self.test_ds)
        if print_res:
            print(f"Test Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        return accuracy, precision, recall, f1, val_loss

    def run(self):
        self.train()
        self.evaluate()
