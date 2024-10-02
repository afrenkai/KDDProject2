import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

class GaussianNbClassifier:
    def __init__(self, train_ds, val_ds, test_ds, unique_styles, max_epoch = 1):
        self.clf = GaussianNB()
        self.unique_styles = unique_styles
        # use preprocessed dataset from ../processed_data
        self.train_ds = train_ds 
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.max_epoch = max_epoch
        self.encoder  = LabelEncoder().fit(unique_styles)
        self.encoded_classes = self.encoder.transform(unique_styles)
        self.clf_name = 'Gaussian NB'

    # untested need to check return type and shape
    def style2label(self, style):
        return self.encoder.transform(style)

    # untested
    def label2style(self, label):
        return self.encoder.inverse_transform(label)

    
    # trains for a single epoch return validation accuracy
    def train(self):
        for batch_num, (x_batch_train, y_batch_train) in enumerate(tqdm(self.train_ds)):
            self.clf.partial_fit(x_batch_train, y_batch_train, classes=self.encoded_classes)
        # print validation set acc after every epoch
        x_val, y_val = self.get_test_val_x_y(self.val_ds)
        y_pred = self.clf.predict(x_val)
        print("Validation set acc:", accuracy_score(y_val, y_pred))

    def get_test_val_x_y(self, ds: Dataset):
        return self.test_ds['img_pixels'], self.test_ds['label']

    def evaluate(self):
        print(f"Evaluating {self.clf_name}")
        x_test, y_test = self.get_test_val_x_y(self.test_ds)
        y_pred = self.clf.predict(x_test)

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        print("Accuracy:")
        print(accuracy_score(y_test, y_pred))



    def run(self):
        print(f"run() called for {self.clf_name}")
        for i in range(self.max_epoch):
            print(f"Training: {i+1}/{self.max_epoch} epochs")
            self.train()
        self.evaluate()
