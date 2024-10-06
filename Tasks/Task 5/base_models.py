from Models.Base_Models.GaussianNaiveBayes import  GaussianNbClassifier
from Models.Base_Models.SVMSGD import SVMClassifier
from Models.Base_Models.XGBoost import XGBoostClassifier
from Models.Base_Models.KNN import KNNImageClassifier
from load_data import get_datasets
from datasets import DatasetDict
from sklearn.metrics import  accuracy_score


if __name__ == '__main__':
    NUM_SAMPLES = None # whole dataset
    # load data like this for batched
    train_dataset_batched, val_dataset, test_dataset, unique_styles = get_datasets(for_CNN=False, val_size=0.2, 
                                                                              batch_size=64, num_samples=NUM_SAMPLES)

    
    # get un-batched dataset
    train_dataset, _, _, _ = get_datasets(for_CNN=False, val_size=0.2,batch_size=None, num_samples=NUM_SAMPLES)


    gnb_clf = GaussianNbClassifier(train_dataset, val_dataset, test_dataset, unique_styles)
    gnb_clf.run()

    svm_clf = SVMClassifier(train_dataset_batched, val_dataset, test_dataset, unique_styles, max_epoch=5, n_jobs=-1)
    svm_clf.run()

    xgb_clf = XGBoostClassifier(train_dataset, val_dataset, test_dataset, unique_styles, n_jobs=-1)
    xgb_clf.run()


    knn_clf = KNNImageClassifier(train_dataset, val_dataset, test_dataset, unique_styles, n_neighbors=20)
    knn_clf.run()
