from Models.Base_Models.GaussianNaiveBayes import  GaussianNbClassifier
from Models.Base_Models.SVMSGD import SVMClassifier
from Models.Base_Models.XGBoost import XGBoostClassifier
from Models.Base_Models.KNN import KNNImageClassifier
from load_data import get_datasets
from datasets import DatasetDict
from sklearn.metrics import  accuracy_score


if __name__ == '__main__':
    NUM_SAMPLES = 7000
    # load data like this for batched
    train_dataset_batched, val_dataset, test_dataset, unique_styles = get_datasets(for_CNN=False, val_size=0.2, 
                                                                              batch_size=32, num_samples=NUM_SAMPLES)

    
    # get un-batched dataset
    train_dataset, _, _, _ = get_datasets(for_CNN=False, val_size=0.2,batch_size=None, num_samples=NUM_SAMPLES)


    # gnb_clf = GaussianNbClassifier(train_dataset_batched, val_dataset, test_dataset, unique_styles, max_epoch=2)
    # gnb_clf.tune()

    # svm_clf = SVMClassifier(train_dataset_batched, val_dataset, test_dataset, unique_styles, max_epoch=2, n_jobs=-1,
    #                         train_unbatched=train_dataset)
    # svm_clf.tune()


    # knn_clf = KNNImageClassifier(train_dataset, val_dataset, test_dataset, unique_styles, n_neighbors=20)
    # knn_clf.tune()


    xgb_clf = XGBoostClassifier(train_dataset, val_dataset, test_dataset, unique_styles, n_jobs=-1)
    xgb_clf.tune()


