from Models.Base_Models.GaussianNaiveBayes import  GaussianNbClassifier
from Models.Base_Models.SVMSGD import SVMClassifier

from load_data import get_tf_datasets
from datasets import DatasetDict
from sklearn.metrics import  accuracy_score


def style2idx(style, style_list: list):
    return style_list.index(style)

def idx2style(idx, style_list):
    return style_list[idx]


if __name__ == '__main__':
    # load data
    train_dataset, val_dataset, test_dataset, unique_styles = get_tf_datasets(for_CNN=False, val_size=0.2, 
                                                                              batch_size=10000)
    print("Element spec", train_dataset.element_spec)
    # gnb_clf = GaussianNbClassifier(train_dataset, val_dataset, test_dataset, unique_styles, max_epoch=2)
    # gnb_clf.run()


    # svm_clf = SVMClassifier(train_dataset, val_dataset, test_dataset, unique_styles, max_epoch=2, n_jobs=-1)
    # svm_clf.run()