


class GaussianNaiveBayes(GaussianNB):
    def __init__(self):
        super().__init__()

    def train(self, x, y):
        pass

    def predict(self, x, y):
        pass

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

class GaussianNbClassifier:
    def __init__(self, img_height=64, img_width=64, batch_size=500, max_iter=1000):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.sgd = GaussianNB()
        self.unique_styles = []
        self.train_ds = load_dataset(self.dataset_name) # use the preprocessing one
        self.val_ds = None
        self.test_ds = self.dataset['train'].select(range(self.n_obs))

    def preprocess_images(self):
        def convert_img(x):
            img_array = np.array(x['image'].resize((self.img_width, self.img_height)))
            if img_array.shape == (self.img_height, self.img_width, self.channels):
                x['img_pixels'] = img_array.reshape(-1) / 255
            else:
                x['img_pixels'] = None
            return x

        self.train_ds = self.train_ds.map(convert_img, num_proc=4)
        self.train_ds = self.train_ds.filter(lambda x: x['img_pixels'] is not None)

    def encode_labels(self):
        self.unique_styles = list(set(self.train_ds['style']))

        def encode_labels(x):
            x['label'] = self.unique_styles.index(x['style'])
            return x

        self.train_ds = self.train_ds.map(encode_labels, num_proc=4)

    def split_data(self):

        X = np.array([img for img in self.train_ds['img_pixels']]) 
        y = np.array(self.train_ds['label'])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        self.sgd.fit(self.X_train, self.y_train)

    def evaluate(self):

        y_pred = self.sgd.predict(self.X_test)


        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

    def run(self):
    
        self.preprocess_images()
        self.encode_labels()
        self.split_data()
        self.train()
        self.evaluate()

if __name__ == '__main__':
    classifier = SGDClassifier(dataset_name="jlbaker361/wikiart", n_obs=5000, alpha=0.0001, max_iter=1000, tol=1e-3)
    classifier.run()
