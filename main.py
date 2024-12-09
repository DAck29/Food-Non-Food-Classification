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

# Set random seed for reproducibility
seed = 42
random.seed(seed)  # Built-in Python random seed
np.random.seed(seed)  # Numpy random seed
torch.manual_seed(seed)  # PyTorch random seed

# Set seed for CUDA operations, if GPU is available
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def train_and_validate(model, train_loader, val_loader, eval_loader, ood_loader, criterion, optimizer, device, num_epochs, results_dir, hyperparams):
    train_losses = []
    val_losses = []
    accuracies = []

    # Initialize the cumulative confusion matrix
    num_classes = len(train_loader.dataset.classes)
    cumulative_conf_matrix = torch.zeros(num_classes, num_classes)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc="Training Progress", leave=True)):
            # Move inputs and labels to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Calculate test loss and accuracy
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                all_predictions.append(predicted)
                all_labels.append(targets)

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        accuracy = 100 * correct / total
        accuracies.append(accuracy)

        evaluate_model_epoch(model,eval_loader, criterion, device, results_dir, epoch_nr=epoch, num_epochs=num_epochs)

        # Initialize dictionaries to store results for different methods
        auroc_results = {}
        fpr_at_95_results = {}
        roc_data_dict = {}


        # Evaluate each method and store results
        for method in ["MSP", "MaxLog", "ODIN"]:
            auroc, fpr_at_95_tpr = OOD.compute_auroc_epoch(model, eval_loader, ood_loader, device, results_dir, method=method, epoch_nr=epoch)
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
        OOD.plot_auroc_curves_epoch(roc_data_dict, results_dir, epoch_nr=epoch)
        OOD.plot_metrics_bar_chart_epoch(auroc_results, fpr_at_95_results, results_dir, epoch_nr=epoch)

        # Compute confusion matrix
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        conf_matrix = results.compute_confusion_matrix(all_predictions, all_labels,
                                                       num_classes=len(train_loader.dataset.classes))
        # Update cumulative confusion matrix
        cumulative_conf_matrix += conf_matrix

        # Plot and save confusion matrix
        results.plot_confusion_matrix(
            conf_matrix=conf_matrix,
            class_names=train_loader.dataset.classes,
            results_dir=results_dir,
            epoch_nr=epoch
        )
        # Plot the cumulative confusion matrix
        results.plot_all_confusion_matrices(
            conf_matrix=cumulative_conf_matrix,
            class_names=train_loader.dataset.classes,
            results_dir=results_dir
        )

        # Save model to Results folder
        results.save_model_results(model, results_dir, epoch)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        metrics_path = os.path.join(results_dir, 'epoch_metrics.txt')
        with open(metrics_path, 'a') as f:
            f.write(
                f"Epoch {epoch + 1}/{num_epochs}, Hyperparameters: {hyperparams}, Train Loss: {train_loss:.4f}, Test Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%\n")

        ood_metrics_path = os.path.join(results_dir, 'ood_metrics.txt')
        with open(ood_metrics_path, 'a') as f:
            f.write(f"Epoch {epoch + 1}/{num_epochs}, Hyperparameters: {hyperparams}\n")
            for method in ["MSP", "MaxLog", "ODIN"]:
                f.write(
                    f"{method} - AUROC: {auroc_results[method]:.4f}, FPR at 95% TPR: {fpr_at_95_results[method]:.4f}\n")
            f.write("\n")
    # Save training results
    results.save_training_results(train_losses, val_losses, num_epochs, accuracies, results_dir)
    results.plot_all_confusion_matrices(cumulative_conf_matrix, train_loader.dataset.classes, results_dir)

    return accuracies

def evaluate_model_epoch(model, eval_loader, criterion, device, results_dir, epoch_nr, num_epochs):
    model.eval()
    eval_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            eval_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Collect predictions and labels
            all_predictions.append(predicted)
            all_labels.append(targets)

    eval_loss = eval_loss / len(eval_loader.dataset)
    accuracy = 100 * correct / total

    # Compute confusion matrix
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    conf_matrix = results.compute_confusion_matrix(all_predictions, all_labels,
                                                   num_classes=len(eval_loader.dataset.classes))

    # Plot and save confusion matrix
    results.plot_evol_confusion_matrix(
        conf_matrix=conf_matrix,
        class_names=eval_loader.dataset.classes,
        results_dir=results_dir
    )

    print(f"Epoch [{epoch_nr + 1}/{num_epochs}], Evaluation Loss: {eval_loss:.4f}, Evaluation Accuracy: {accuracy:.2f}%")
    metrics_path = os.path.join(results_dir, 'evaluation_metrics.txt')
    with open(metrics_path, 'a') as f:
        f.write(f"Epoch {epoch_nr + 1}/{num_epochs}, Evaluation Loss: {eval_loss:.4f}, Evaluation Accuracy: {accuracy:.2f}%\n")


def evaluate_model(model, eval_loader, criterion, device, results_dir):
    model.eval()
    eval_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            eval_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Collect predictions and labels
            all_predictions.append(predicted)
            all_labels.append(targets)

    eval_loss = eval_loss / len(eval_loader.dataset)
    accuracy = 100 * correct / total

    # Compute confusion matrix
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    conf_matrix = results.compute_confusion_matrix(all_predictions, all_labels,
                                                   num_classes=len(eval_loader.dataset.classes))

    # Plot and save confusion matrix
    results.plot_evol_confusion_matrix(
        conf_matrix=conf_matrix,
        class_names=eval_loader.dataset.classes,
        results_dir=results_dir
    )

    print(f"Evaluation Loss: {eval_loss:.4f}, Evaluation Accuracy: {accuracy:.2f}%")
    metrics_path = os.path.join(results_dir, 'evaluation_metrics.txt')
    with open(metrics_path, 'a') as f:
        f.write(f"Evaluation Loss: {eval_loss:.4f}, Evaluation Accuracy: {accuracy:.2f}%\n")


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

    base_results_dir = '/storage/homefs/da17u029/DD_DM/Food-Non-Food-Classification/Results'
    #base_results_dir = 'Results'
    # base_results_dir = '/storage/homefs/da17u029/DD_DM/Food-Non-Food-Classification/Results'
    #base_results_dir = '/storage/homefs/ma20e073/FoodClassifierScript/Results'
    # base_results_dir = r"C:\Users\manu_\OneDrive - Universitaet Bern\03 HS24 UniBe-VIVO\05 Diabetes Management\GitHub_Clone\Food-Non-Food-Classification-1\Results"

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

        accuracies = train_and_validate(model, train_loader, val_loader, eval_loader, ood_loader, criterion, optimizer, device, num_epochs, hyperparam_results_dir, hyperparams=params)

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

    # Evaluate the model on the evaluation set
    evaluate_model(model, eval_loader, criterion, device, results_dir)

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