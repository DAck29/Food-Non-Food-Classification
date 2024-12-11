import random

from sklearn.utils import compute_class_weight
from sklearn.metrics import roc_curve, roc_auc_score
from data_distribution import plot_data_distribution, plot_total_images


from data_loader import get_dataloader, get_ood_loader
from resnet_model import get_resnet50_model
from tqdm import tqdm
import OOD
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import results
from datetime import datetime
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import ParameterGrid
from train_evaluate import train_and_evaluate

# Set random seed for reproducibility
seed = 42
random.seed(seed)  # Built-in Python random seed
np.random.seed(seed)  # Numpy random seed
torch.manual_seed(seed)  # PyTorch random seed

# Set seed for CUDA operations, if GPU is available
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def main():
    # Initialize DataLoader
    train_loader, val_loader, eval_loader = get_dataloader(batch_size=32)

    # Evaluate AUROC for MSP, MaxLog, and ODIN
    ood_loader = get_ood_loader(batch_size=32, num_samples=len(
        eval_loader.dataset))  # Make sure that the same Nr of ID and OOD samples are processed

    # Set the number of classes based on your dataset
    num_classes = len(train_loader.dataset.classes)  # Automatically set based on dataset
    model = get_resnet50_model(num_classes=num_classes, pretrained=True)

    # Check if GPU is available and move model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Print model summary
    print(model)

    # Calculate class weights
    train_labels = train_loader.dataset.targets
    classes = np.unique(train_labels)  # Get unique class labels
    class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)  # Move to GPU

    # Plot data distribution
    # plot_data_distribution(train_loader, dataset_type="Training")
    # plot_total_images(train_loader, val_loader, eval_loader, dataset_type="Food Dataset")


    # Define training parameters
    num_epochs = 20
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    param_grid = {
        'lr': [0.001, 0.01],
        'momentum': [0.9, 0.95]
    }

    best_accuracy = 0
    best_params = None
    accuracies = []

    # relative path
    current_directory = os.path.dirname(os.path.abspath(__file__))
    base_results_dir = os.path.join(current_directory, "Results")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(base_results_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    for params in ParameterGrid(param_grid):
        # Create subfolder for each hyperparameter combination
        subfolder_name = f"lr_{params['lr']}_momentum_{params['momentum']}"
        hyperparam_results_dir = os.path.join(results_dir, subfolder_name)
        os.makedirs(hyperparam_results_dir, exist_ok=True)

        print(f"Training with parameters: {params}")
        optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])

        accuracies = train_and_evaluate(model, train_loader, val_loader, eval_loader, ood_loader, criterion, optimizer, device, num_epochs, hyperparam_results_dir, hyperparams=params)

        # Evaluate on validation set
        if accuracies:
            accuracy = accuracies[-1]  # Get the last recorded accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
                best_model_path = os.path.join(results_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)

    print(f"Best Parameters: {best_params}, Best Accuracy: {best_accuracy:.2f}%")

    best_params_file = os.path.join(results_dir, 'best_hyperparameters.txt')
    with open(best_params_file, 'w') as f:
        f.write(f"Best Parameters: {best_params}, Best Accuracy: {best_accuracy:.2f}%\n")


    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint)

    # Initialize dictionaries to store results for different methods
    auroc_results = {}
    fpr_at_95_results = {}
    roc_data_dict = {}

    # Evaluate each method and store results
    for method in ["MSP", "MaxLog", "ODIN"]:
        auroc, fpr_at_95_tpr = OOD.compute_auroc(model, eval_loader, ood_loader, device, results_dir, method=method)
        auroc_results[method] = auroc
        fpr_at_95_results[method] = fpr_at_95_tpr
        print(f"AUROC ({method}): {auroc:.4f}")
        print(f"FPR at 95% TPR ({method}): {fpr_at_95_tpr:.4f}")
        # Calculate FPR and TPR for combined plot and store them in roc_data_dict
        scores = np.array([1] * len(eval_loader.dataset) + [0] * len(ood_loader.dataset))
        labels = np.array([1] * len(eval_loader.dataset) + [0] * len(ood_loader.dataset))
        fpr, tpr, _ = roc_curve(labels, scores)

        roc_data_dict[method] = (fpr, tpr, auroc)

    # Plot the AUROC and FPR@95TPR as bar charts
    OOD.plot_auroc_curves(roc_data_dict, results_dir)
    OOD.plot_metrics_bar_chart(auroc_results, fpr_at_95_results, results_dir)



if __name__ == '__main__':
    main()