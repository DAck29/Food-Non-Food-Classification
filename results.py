import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

def compute_confusion_matrix(predictions, labels, num_classes):
    """
        Compute the confusion matrix.

        Args:
            predictions: prediction matrix.
            labels: labels to be evaluated.
            num_classes: Nr. of classes.

        Returns:
            conf_matrix: confusion matrix.
    """
    # Initialize the confusion matrix with zeros
    conf_matrix = torch.zeros(num_classes, num_classes)

    # Iterate over predictions and true labels
    for t, p in zip(labels.view(-1), predictions.view(-1)):
        conf_matrix[t.long(), p.long()] += 1

    return conf_matrix

def plot_confusion_matrix(conf_matrix, class_names, results_dir: str, epoch_nr: int = None, title='Confusion Matrix'):
    """
        This function prints and plots the confusion matrix.

        Args:
            conf_matrix: confusion matrix.
            class_names: names of the classes.
            results_dir: directory to save the figure.
            epoch_nr: Nr. of epochs.
            title: title of the plot.

        Returns:
            None
    """

    plt.figure()
    sns.heatmap(conf_matrix.numpy(), annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.tight_layout()

    # Generate file path based on the presence of epoch_nr
    if epoch_nr is not None:
        confusion_matrix_path = os.path.join(results_dir, f"confusion_matrix_epoch_{epoch_nr + 1}.png")
    else:
        confusion_matrix_path = os.path.join(results_dir, "confusion_matrix.png")

    plt.savefig(confusion_matrix_path)
    plt.close()

def plot_auroc_curve(fpr, tpr, auroc, method, results_dir, epoch_nr=None):
    """
        Plots the AUROC curve and saves it to the results directory.

        Args:
            fpr: False positive rates.
            tpr: True positive rates.
            auroc: AUROC score.
            method: Method used for computation.
            results_dir: Directory to save the plot.
            epoch_nr: Epoch number.

        Returns:
            None
    """
    plt.figure()
    plt.plot(fpr, tpr, label=f'{method} AUROC: {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')  # Random guess line
    plt.xlabel('False Positive (In-Distribution) Rate', fontsize=14)
    plt.ylabel('True Positive (In-Distribution) Rate', fontsize=14)
    plt.title(f'Receiver Operating Characteristic (ROC) Curve ({method})', fontsize=16)
    plt.legend()
    plt.grid()

    # Save plot in the current folder
    auroc_filename = f"auroc_{method.lower()}_epoch_{epoch_nr}_plot.png" if epoch_nr is not None else f"auroc_{method.lower()}_plot.png"
    auroc_path = os.path.join(results_dir, auroc_filename)
    plt.savefig(auroc_path)
    print(f"AUROC plot saved to {auroc_path}")
    plt.close()

def plot_metrics_bar_chart(auroc_dict, fpr_at_95_dict, results_dir, epoch_nr=None):
    """
        Plots the metrics bar chart and saves it to the results directory.

        Args:
            auroc_dict: AUROC score.
            fpr_at_95_dict: FPR score.
            results_dir: Directory to save the plot.
            epoch_nr: Epoch number.

        Returns:
            None
    """
    # Plot AUROC as a bar chart
    methods = list(auroc_dict.keys())
    auroc_values = list(auroc_dict.values())

    plt.figure()
    plt.bar(methods, auroc_values, color='skyblue')
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('AUROC', fontsize=14)
    plt.title('AUROC for Different Methods', fontsize=16)
    plt.ylim([0, 1])
    plt.grid(axis='y', linestyle='--')

    auroc_bar_filename = f"auroc_bar_chart_{epoch_nr}.png" if epoch_nr is not None else 'auroc_bar_chart.png'
    auroc_bar_path = os.path.join(results_dir, auroc_bar_filename)
    plt.savefig(auroc_bar_path)
    print(f"AUROC bar chart saved to {auroc_bar_path}")
    plt.close()

    # Plot FPR@95TPR as a bar chart
    fpr_at_95_values = list(fpr_at_95_dict.values())

    plt.figure()
    plt.bar(methods, fpr_at_95_values, color='salmon')
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('FPR at 95% TPR', fontsize=14)
    plt.title('FPR at 95% TPR for Different Methods', fontsize=16)
    plt.ylim([0, 1])
    plt.grid(axis='y', linestyle='--')

    fpr_at_95_bar_filename = f"fpr_at_95_tpr_bar_chart_{epoch_nr}.png" if epoch_nr is not None else 'fpr_at_95_tpr_bar_chart.png'
    fpr_bar_path = os.path.join(results_dir, fpr_at_95_bar_filename)
    plt.savefig(fpr_bar_path)
    print(f"FPR at 95% TPR bar chart saved to {fpr_bar_path}")
    plt.close()

def save_model_results(model: nn.Module, results_dir: str, epoch_nr: int) -> None:
    """
        This function saves the model and its results.

        Args:
            model: model to be saved.
            results_dir: directory to save the figures and model results.
            epoch_nr: Nr. of epochs.

        Returns:
            None
    """

    model_path = os.path.join(results_dir, f"resnet50_model_{epoch_nr+1}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def save_training_results(train_losses, test_losses, num_epochs,accuracies, dir_results):
    """
        This function saves the training results.

        Args:
            train_losses: training losses.
            test_losses: test losses.
            num_epochs: number of epochs.
            accuracies: accuracies.
            dir_results: directory to save the figures and model results.

        Returns:
            None
    """

    # Create folder with timestamp to track tests
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(dir_results, exist_ok=True)
    results_path = os.path.join(dir_results, f"training_results_{timestamp}.txt")
    smoothed_train_losses = gaussian_filter1d(train_losses, sigma=7)

    # write training losses in a .txt file that was created
    with open(results_path, "w") as f:
        f.write("Training Losses:\n")
        for i, loss in enumerate(smoothed_train_losses):
            f.write(f"Iteration {i + 1}: Loss = {loss:.4f}\n")

    print(f"Training results saved to {results_path}")

    # Create plot with training loss and save it
    plot_path = os.path.join(dir_results, f"training_loss_plot_{timestamp}.png")
    plt.figure()
    plt.plot(smoothed_train_losses)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Cross Entropy Loss', fontsize=16)
    plt.grid()
    plt.savefig(plot_path)
    print(f"Training loss plot saved to {plot_path}")

    # Create plot with loss curve and save it
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

    # Create plot with accuracy and save it
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


