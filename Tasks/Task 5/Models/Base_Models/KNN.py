import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from sklearn.model_selection import ParameterGrid, GridSearchCV
import warnings
warnings.filterwarnings("ignore")

class KNNImageClassifier:
    def __init__(self, train_ds, val_ds, test_ds, unique_styles, n_neighbors=20):
        self.n_neighbors = n_neighbors
        self.clf = KNeighborsClassifier(n_neighbors=self.n_neighbors,n_jobs=-1)
        self.unique_styles = unique_styles
        self.train_ds = train_ds 
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.encoder  = LabelEncoder().fit(unique_styles)
        self.encoded_classes = self.encoder.transform(unique_styles)
        self.clf_name = 'KNN'

    def train(self):
        x_train, y_train = self.get_test_val_x_y(self.train_ds)
        self.clf.fit(x_train, y_train)
        x_val, y_val = self.get_test_val_x_y(self.val_ds)
        y_pred = self.clf.predict(x_val)
        print("Validation set acc:", accuracy_score(y_val, y_pred))

    def get_test_val_x_y(self, ds: Dataset):
        return ds['img_pixels'], ds['label']


    def evaluate(self):
        print(f"Evaluating {self.clf_name}")
        x_test, y_test = self.get_test_val_x_y(self.test_ds)
        y_pred = self.clf.predict(x_test)

        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.unique_styles))
        
        print("Accuracy:")
        print(accuracy_score(y_test, y_pred))

    def tune(self):
        print(f"Tuning hyperparams for {self.clf_name}")
        param_grid = [
            {'n_neighbors': [20, 50, 100]}
        ]
        print("Grid Search on Validation set")
        best_score = -1
        best_grid = None
        for g in ParameterGrid(param_grid):
            self.clf.set_params(**g)
            x_val, y_val = self.get_test_val_x_y(self.val_ds)
            self.clf.fit(x_val, y_val)
            y_pred = self.clf.predict(x_val)
            current_score = accuracy_score(y_val, y_pred)
            # save if best
            if current_score > best_score:
                best_score = current_score
                best_grid = g
        print(f"Best parameters: {best_grid} with accuracy: {best_score}")
        self.clf.set_params(**best_grid)
        self.evaluate()


        # cross validation
        print("Grid Search with CV K-fold=3")
        x_train, y_train = self.get_test_val_x_y(self.train_ds)
        grid_search = GridSearchCV(
            estimator=self.clf, param_grid=param_grid, cv=3, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print(f"Best parameters: {grid_search.best_params_} with accuracy: {grid_search.best_score_}")
        self.clf = grid_search.best_estimator_
        self.evaluate()

    def run(self):
        print(f"run() called for {self.clf_name}")
        self.train()
        self.evaluate()

