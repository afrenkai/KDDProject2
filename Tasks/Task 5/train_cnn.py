from Models.Base_Models.CNN import CNNClassifier
from load_data import get_datasets
from datasets import DatasetDict
from sklearn.metrics import  accuracy_score


if __name__ == '__main__':
    NUM_SAMPLES = None
    # load data like this for batched
    train_dataset_batched, val_dataset, test_dataset, unique_styles = get_datasets(for_CNN=True, val_size=0.1, 
                                                                              batch_size=128, num_samples=NUM_SAMPLES)

    # # get un-batched dataset
    # train_dataset, _, _, _ = get_datasets(for_CNN=False, val_size=0.2,batch_size=None, num_samples=NUM_SAMPLES)

    cnn_clf = CNNClassifier(train_dataset_batched, val_dataset, test_dataset, unique_styles, learning_rate=0.001,
                            dropout_rate=0.1, epochs=20)
    cnn_clf.run()




