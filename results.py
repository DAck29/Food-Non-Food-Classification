import torch
import os
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def compute_confusion_matrix(predictions, labels, num_classes):
    # Initialize the confusion matrix with zeros
    conf_matrix = torch.zeros(num_classes, num_classes)

    # Iterate over predictions and true labels
    for t, p in zip(labels.view(-1), predictions.view(-1)):
        conf_matrix[t.long(), p.long()] += 1

    return conf_matrix

def plot_confusion_matrix(conf_matrix, class_names, results_dir, epoch_nr, title='Confusion Matrix'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix.numpy(), annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()

    confusion_matrix_path = os.path.join(results_dir, f"confusion_matrix_epoch_{epoch_nr + 1}.png")
    plt.savefig(confusion_matrix_path)
    plt.close()


def save_model_results(model: nn.Module, results_dir: str, epoch_nr: int) -> None:
    model_path = os.path.join(results_dir, f"resnet50_model_{epoch_nr+1}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")