from Models.Base_Models.CNN import CNNClassifier
from load_data import get_datasets, to_pytorch_dataloader, load_data
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import itertools
import numpy as np
import torch

NUM_SAMPLES = 20000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hyperparameter_tuning(hyperparams):
    print('Starting Grid Search')
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
        
        cnn_clf.train()

        accuracy, precision, recall, f1, val_loss = cnn_clf.evaluate(False)
        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, Validation Loss: {val_loss}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

        print(f"Best accuracy so far: {best_accuracy} with params {best_params}")

    print(f"Best parameters: {best_params} with accuracy: {best_accuracy}")
    return best_params

def k_fold_cross_validation(params, k=5):

    train_dataset, _, test_dataset, unique_styles = load_data(
        for_CNN=True, 
        val_size=0.5,  
        num_samples=NUM_SAMPLES,
    )

    test_dataset = to_pytorch_dataloader(test_dataset, batch_size=params['batch_size'])

    X = np.zeros(train_dataset.num_rows)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    val_losses = []


    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{k}")
        
        train_dataset_hf = train_dataset.select(train_idx, keep_in_memory=True)

        val_dataset_hf = train_dataset.select(val_idx, keep_in_memory=True)

        val_dataset_hf.set_format(type='torch', columns=['img_pixels', 'label'], device=device)
        train_dataset_batched = to_pytorch_dataloader(train_dataset_hf, batch_size=params['batch_size'])
        
        cnn_clf = CNNClassifier(
            train_dataset_batched, 
            val_dataset_hf, 
            test_dataset, 
            unique_styles, 
            batch_size=params['batch_size'], 
            epochs=params['epochs'], 
            learning_rate=params['learning_rate'],
            dropout_rate=params['dropout_rate']
        )

        cnn_clf.train()

        accuracy, precision, recall, f1, val_loss = cnn_clf.evaluate(False)
        print(f"Fold {fold + 1} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, Validation Loss: {val_loss}")

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        val_losses.append(val_loss)

    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1_score = np.mean(f1_scores)
    avg_val_loss = np.mean(val_losses)

    print(f"Average K-fold metrics: \n"
          f"Accuracy: {avg_accuracy}\n"
          f"Precision: {avg_precision}\n"
          f"Recall: {avg_recall}\n"
          f"F1 Score: {avg_f1_score}\n"
          f"Validation Loss: {avg_val_loss}")

    return avg_accuracy, avg_precision, avg_recall, avg_f1_score, avg_val_loss

def hyperparameter_tuning_with_k_fold(hyperparams, k=5):
    print(f'Starting Grid Search + CV (k={k})')
    best_accuracy = 0.0
    best_params = None

    for params in hyperparams:
        print(f"Testing with params: {params}")

        avg_accuracy, _,_, _,_ = k_fold_cross_validation(params, k=k)

        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_params = params

        print(f"Best accuracy so far: {best_accuracy} with params {best_params}")

    print(f"Best parameters: {best_params} with accuracy: {best_accuracy}")
    return best_params

if __name__ == '__main__':
    param_grid = {
        'batch_size': [16, 32, 64],
        'epochs': [1, 3, 5, 10],
        'learning_rate': [0.001, 0.0001],
        'dropout_rate': [0.0, 0.1, 0.2]
    }


    hyperparams = [
        dict(zip(param_grid.keys(), values)) 
        for values in itertools.product(*param_grid.values())
    ]

    # best_hyperparams_standard = hyperparameter_tuning(hyperparams)
    # print('Grid Search Best params')
    # print(best_hyperparams_standard)

    best_hyperparams_kfold = hyperparameter_tuning_with_k_fold(hyperparams, k=3)
    print('Grid Search + CV (k=3)')
    print(best_hyperparams_kfold)
