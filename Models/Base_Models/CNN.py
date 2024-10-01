import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_dataset, load_from_disk

class ImgDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0,3,1,2) #reshaping for channels
        self.y = torch.tensor(y, dtype = torch.long)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class CNN(nn.Module):
    def __init__(self, num_classes, img_channels=3):
        super (CNN, self).__init__()
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128 * 8 * 8, 128) #works under the assumption that image is of size 64 x 64
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8) # flatten here
        x = torch.relu(self.fc1(x)) #1st fully connected layer
        x = self.fc2(x)
        return x
    
class CNNClassifier():
    def __init__(self, dataset_name, img_height = 64, img_width = 64, channels = 3, n_obs = 5000, batch_size = 32, epochs = 10, learning_rate = 0.001):
        self.dataset_name = dataset_name
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.n_obs = n_obs
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unique_styles = [] #blank list to store each style found in ds

        self.dataset = load_dataset(self.dataset_name)
        self.train_ds = self.dataset['train'].select(range(self.n_obs))

    def preprocess(self):
        def convert_img(x):
            img_arr = np.array(x['image'].resize((self.img_width, self.img_height)))
            if img_arr.shape == (self.img_height, self.img_width, self.channels):
                x['img_pixels'] = img_arr / 255.0
            else:
                x['img_pixels'] = None
            return x
        self.train_ds = self.train_ds.map(convert_img, num_proc=6)
        self.train_ds = self.train_ds.filter(lambda x: x['img_pixels'] is not None)

    def encode_labels(self):
        self.unique_styles = list(set(self.train_ds['style']))

        def encode(x):
            x['label'] = self.unique_styles.index(x['style'])
            return x

        self.train_ds = self.train_ds.map(encode, num_proc= 6)

    
    def split_data(self):
        X = np.array([img for img in self.train_ds['img_pixels']]) #3d shenanigans
        y = np.array(self.train_ds['label'])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size= 0.2, random_state= 69)

        train_dataset = ImgDataset(self.X_train, self.y_train)
        test_dataset = ImgDataset(self.X_test, self.y_test)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle = True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle = False)

    def build_model(self):
        self.model = CNN(num_classes = len(self.unique_styles)).to(self.device)

    def train(self):
        self.build_model()

        criterion = nn.CrossEntropyLoss
        optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)

        for epoch in range (self.epochs):
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            print(f'Epoch [{epoch + 1} / {self.epochs}], Loss: {running_loss/len(self.train_loader):.4f}')

    def eval(self):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                mages, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds))

    def run(self):
        self.preprocess()
        self.encode_labels()
        self.split_data()
        self.train()
        self.eval()


if __name__ == '__main__':
    classifier = CNNClassifier(dataset_name= "jlbaker361/wikiart", n_obs= 1000)
    classifier.run()