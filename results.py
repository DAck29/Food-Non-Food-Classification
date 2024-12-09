import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

def compute_confusion_matrix(predictions, labels, num_classes):
    # Initialize the confusion matrix with zeros
    conf_matrix = torch.zeros(num_classes, num_classes)

    # Iterate over predictions and true labels
    for t, p in zip(labels.view(-1), predictions.view(-1)):
        conf_matrix[t.long(), p.long()] += 1

    return conf_matrix

def plot_confusion_matrix(conf_matrix, class_names, results_dir: str, epoch_nr: int, title='Confusion Matrix'):
    plt.figure()
    sns.heatmap(conf_matrix.numpy(), annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.tight_layout()

    confusion_matrix_path = os.path.join(results_dir, f"confusion_matrix_epoch_{epoch_nr + 1}.png")
    plt.savefig(confusion_matrix_path)
    plt.close()

def plot_evol_confusion_matrix(conf_matrix, class_names, results_dir: str, title='Confusion Matrix'):
    plt.figure()
    sns.heatmap(conf_matrix.numpy(), annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.tight_layout()

    confusion_matrix_path = os.path.join(results_dir, f"confusion_matrix_epoch_evaluation.png")
    plt.savefig(confusion_matrix_path)
    plt.close()

def plot_all_confusion_matrices(conf_matrix, class_names, results_dir: str, title='Cumulative Confusion Matrix'):
    plt.figure()
    sns.heatmap(conf_matrix.numpy(), annot=True, fmt='g', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.tight_layout()

    all_conf_matrix_path = os.path.join(results_dir, 'cumulative_confusion_matrices.png')
    plt.savefig(all_conf_matrix_path)
    plt.close()

def save_model_results(model: nn.Module, results_dir: str, epoch_nr: int) -> None:
    model_path = os.path.join(results_dir, f"resnet50_model_{epoch_nr+1}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def save_training_results(train_losses, test_losses, num_epochs,accuracies, dir_results):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(dir_results, exist_ok=True)
    results_path = os.path.join(dir_results, f"training_results_{timestamp}.txt")
    smoothed_train_losses = gaussian_filter1d(train_losses, sigma=7)

    with open(results_path, "w") as f:
        f.write("Training Losses:\n")
        for i, loss in enumerate(smoothed_train_losses):
            f.write(f"Iteration {i + 1}: Loss = {loss:.4f}\n")

    print(f"Training results saved to {results_path}")

    plot_path = os.path.join(dir_results, f"training_loss_plot_{timestamp}.png")
    plt.figure()
    plt.plot(smoothed_train_losses)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Cross Entropy Loss', fontsize=16)
    plt.grid()
    plt.savefig(plot_path)
    print(f"Training loss plot saved to {plot_path}")

    # Save the loss curves
    plot_path = os.path.join(dir_results, f"loss_curve{timestamp}.png")
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.savefig(plot_path)
    print("Loss curve saved to loss_curve.png")

    plot_path = os.path.join(dir_results, f"accuracy_curve{timestamp}.png")
    plt.figure()
    plt.plot(range(1, num_epochs + 1), accuracies, label='Accuracy')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.savefig(plot_path)
    print("Accuracy curve saved to accuracy_curve.png")
    plt.close()

