import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

class KNNImageClassifier:
    def __init__(self, dataset_name, img_height=64, img_width=64, channels=3, n_obs=10000, n_neighbors=5):
        self.dataset_name = dataset_name
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.n_obs = n_obs
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.unique_styles = []
        
        self.dataset = load_dataset(self.dataset_name)
        self.train_ds = self.dataset['train'].select(range(self.n_obs))

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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=17)

    def train(self):
        self.knn.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.knn.predict(self.X_test)

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
    classifier = KNNImageClassifier(dataset_name="jlbaker361/wikiart", n_obs=40000, n_neighbors=5) 
    classifier.run()
