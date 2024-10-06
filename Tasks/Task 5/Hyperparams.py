from Models.Base_Models.CNN import CNNClassifier
from load_data import get_datasets
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools

def hyperparameter_tuning(hyperparams):
    NUM_SAMPLES = None
    best_accuracy = 0.0
    best_params = None

    for params in hyperparams:
        print(f"Testing with params: {params}")

        train_dataset_batched, val_dataset, test_dataset, unique_styles = get_datasets(
            for_CNN=True, 
            val_size=0.2, 
            batch_size=params['batch_size'], 
            num_samples=NUM_SAMPLES
        )

        cnn_clf = CNNClassifier(
            train_dataset_batched, 
            val_dataset, 
            test_dataset, 
            unique_styles, 
            batch_size=params['batch_size'], 
            epochs=params['epochs'], 
            learning_rate=params['learning_rate'],
            dropout_rate=params['dropout_rate'] 
        )
        
        cnn_clf.run()

        accuracy, precision, recall, f1, val_loss = cnn_clf.evaluate()

        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, Validation Loss: {val_loss}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

        print(f"Best accuracy so far: {best_accuracy} with params {best_params}")

    print(f"Best parameters: {best_params} with accuracy: {best_accuracy}")
    return best_params

if __name__ == '__main__':
    param_grid = {
        'batch_size': [32, 64],
        'epochs': [5, 10],
        'learning_rate': [0.001, 0.0001],
        'dropout_rate': [0.0, 0.2]  
    }

    hyperparams = [
        dict(zip(param_grid.keys(), values)) 
        for values in itertools.product(*param_grid.values())
    ]
    best_hyperparams = hyperparameter_tuning(hyperparams)
