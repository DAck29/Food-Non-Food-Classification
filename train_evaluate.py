#import torch
#from openpyxl.styles.builtins import title
from sklearn.metrics import roc_curve
from tqdm import tqdm
import OOD
import torch
import results
import os
import numpy as np


def train_and_evaluate(model, train_loader, val_loader, eval_loader, ood_loader, criterion, optimizer, device, num_epochs, results_dir, hyperparams):
    """
    Train, validate and evaluate a model with different Hyperparameters and save results.
    
    Args:
        model: The model to be trained and evaluated.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        eval_loader: DataLoader for the evaluation dataset.
        ood_loader: DataLoader for the out-of-distribution cifar10 dataset.
        criterion: Loss function.
        optimizer: Optimizer for training.
        device: Device to use for training ("cuda" or "cpu").
        num_epochs: Nr of training epochs.
        results_dir: Directory to save results
        hyperparams: Hyperparameter.

    Returns:
        list: List of accuracies for each epoch.
    """ 
    
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

        evaluate_model(model,eval_loader, criterion, device, results_dir, epoch_nr=epoch, num_epochs=num_epochs)

        # Initialize dictionaries to store results for different methods
        auroc_results = {}
        fpr_at_95_results = {}
        roc_data_dict = {}


        # Evaluate each method and store results
        for method in ["MSP", "MaxLog", "ODIN"]:
            auroc, fpr_at_95_tpr = OOD.compute_auroc_fpr95(model, eval_loader, ood_loader, device, results_dir, method=method, epoch_nr=epoch)
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
        results.plot_auroc_curve(roc_data_dict, results_dir, epoch_nr=epoch)
        results.plot_metrics_bar_chart(auroc_results, fpr_at_95_results, results_dir, epoch_nr=epoch)

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

    return accuracies

def evaluate_model(model, eval_loader, criterion, device, results_dir, epoch_nr, num_epochs):
    """
    Evaluate a model on evaluation dataset and save results.

    Args:
        model: model to be evaluated.
        eval_loader: DataLoader for the evaluation dataset.
        criterion: Loss function.
        device: Device to use ("cuda" or "cpu").
        results_dir: Directory to save results.
        epoch_nr:  current epoch number.
        num_epochs: total number of epochs.

    Returns:
        None
    """

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
    results.plot_confusion_matrix(
        conf_matrix=conf_matrix,
        class_names=eval_loader.dataset.classes,
        results_dir=results_dir,
        epoch_nr=epoch_nr,
        title="Confusion Matrix in Evaluation"
    )

    print(f"Epoch [{epoch_nr + 1}/{num_epochs}], Evaluation Loss: {eval_loss:.4f}, Evaluation Accuracy: {accuracy:.2f}%")
    metrics_path = os.path.join(results_dir, 'evaluation_metrics.txt')
    with open(metrics_path, 'a') as f:
        f.write(f"Epoch {epoch_nr + 1}/{num_epochs}, Evaluation Loss: {eval_loss:.4f}, Evaluation Accuracy: {accuracy:.2f}%\n")


