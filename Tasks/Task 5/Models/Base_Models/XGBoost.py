import xgboost as xgb
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from FLAGS import RANDOM_STATE
from sklearn.model_selection import ParameterGrid, GridSearchCV
import warnings
warnings.filterwarnings("ignore")


# https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
class XGBoostClassifier:
    def __init__(self, train_ds, val_ds, test_ds, unique_styles, 
                 n_estimators=1, max_depth=6, reg_alpha=0, reg_lambda=1,
                 learning_rate=0.3, subsample=1, n_jobs=-1,
                 partial_fit=True, device='cuda', max_epoch=1):
        # XGBClassifier is an interface for skelarn models
        # There is a lower level one that might be a better fit (easier to tune)
        self.clf = xgb.XGBClassifier(
            objective='multi:softmax', n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, n_jobs=None, subsample=subsample,
            reg_lambda=reg_lambda, reg_alpha=reg_alpha, enable_categorical=True,
            random_state=RANDOM_STATE, device=device
            )
        self.unique_styles = unique_styles
        # use preprocessed dataset from ../processed_data
        self.train_ds = train_ds 
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.encoder  = LabelEncoder().fit(unique_styles)
        self.encoded_classes = self.encoder.transform(unique_styles)
        self.clf_name = 'XGBoostClassifier'

    # untested need to check return type and shape
    def style2label(self, style):
        return self.encoder.transform(style)

    # untested
    def label2style(self, label):
        return self.encoder.inverse_transform(label)

    # trains for a single epoch return validation accuracy
    def train(self):
        x_train, y_train = self.get_test_val_x_y(self.train_ds)
        self.clf.fit(x_train, y_train)
        x_val, y_val = self.get_test_val_x_y(self.val_ds)
        y_pred = self.clf.predict(x_val)
        print("Validation set acc:", accuracy_score(y_val, y_pred))

   
    def tune(self):
        print(f"Tuning hyperparams for {self.clf_name}")
        param_grid = [
            {'n_estimators': [1, 5, 10],
             'learning_rate': [0.3, 0.1],
             'subsample': [1, 0.7],
             'reg_lambda': [0.9, 1],
             'reg_alpha': [0, 0.01],
             }
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
            estimator=self.clf, param_grid=param_grid, cv=3, n_jobs=1)
        grid_search.fit(x_train, y_train)
        print(f"Best parameters: {grid_search.best_params_} with accuracy: {grid_search.best_score_}")
        self.clf = grid_search.best_estimator_
        self.evaluate()

    def get_test_val_x_y(self, ds: Dataset):
        return ds['img_pixels'], ds['label']

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
        self.train()
        self.evaluate()
