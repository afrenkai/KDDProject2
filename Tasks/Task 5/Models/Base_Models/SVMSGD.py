import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

class SVMClassifier:
    def __init__(self, train_ds, val_ds, test_ds, unique_styles, alpha=0.0001, 
                 max_epoch=5, tol=1e-3, lr=1e-3, penalty=None, n_jobs=1):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.alpha = alpha
        self.max_epoch = max_epoch
        self.tol = tol
        self.lr = lr
        self.penalty = penalty
        # TODO how to set learning rate here? sklearn only has 3 string options
        # eg. optimal, adaptive etc
        self.clf = SGDClassifier(alpha=self.alpha, tol=self.tol, loss='hinge'
                                 , penalty=self.penalty, n_jobs=n_jobs, random_state=1234) # hinge = SVM
        self.unique_styles = unique_styles
        self.encoder  = LabelEncoder().fit(unique_styles)
        self.encoded_classes = self.encoder.transform(unique_styles)
        self.clf_name = 'SVM+SGD'


    # trains for a single epoch return validation accuracy
    def train(self):
        for batch_num, data in enumerate(self.train_ds):
            y_batch_train = data['label']
            x_batch_train = data['img_pixels']
            self.clf.partial_fit(x_batch_train, y_batch_train, classes=self.encoded_classes)
        # print validation set acc after every epoch
        x_val, y_val = self.get_test_val_x_y(self.val_ds)
        y_pred = self.clf.predict(x_val)
        print("Validation set acc:", accuracy_score(y_val, y_pred))

    def tune_hyperparams():
        '''
        options:
            - grid-search [on validation set]
            - grid-search+CV [on train set (one epoch)]
            - Random Search [on validation]
            - Hyperband
            - Bayesian 
        
        '''
        pass

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
        for i in range(self.max_epoch):
            print(f"Training: {i+1}/{self.max_epoch} epochs")
            self.train()
        self.evaluate()


