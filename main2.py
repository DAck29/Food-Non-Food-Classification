import random

from sklearn.utils import compute_class_weight
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
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import ParameterGrid

from pytorch_ood.detector import TemperatureScaling, ODIN
from sklearn.metrics import roc_auc_score, roc_curve
from OOD import compute_odin_scores

# Set random seed for reproducibility
seed = 42
random.seed(seed)  # Built-in Python random seed
np.random.seed(seed)  # Numpy random seed
torch.manual_seed(seed)  # PyTorch random seed

# Set seed for CUDA operations, if GPU is available
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def save_training_results(losses, train_losses, test_losses, num_epochs, accuracies, dir_results):
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
    plt.figure(figsize=(12, 4))
    plt.plot(smoothed_train_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Cross Entropy Loss')
    plt.grid()
    plt.savefig(plot_path)
    print(f"Training loss plot saved to {plot_path}")

    # Save the loss curves
    plot_path = os.path.join(dir_results, f"loss_curve_{timestamp}.png")
    plt.figure(figsize=(12, 4))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(plot_path)
    print(f"Loss curve saved to loss_curve.png")

    plot_path = os.path.join(dir_results, f"accuracy_curve_{timestamp}.png")
    plt.figure(figsize=(12, 4))
    plt.plot(range(1, num_epochs + 1), accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()
    plt.savefig(plot_path)
    print(f"Accuracy curve saved to accuracy_curve.png")
    plt.close()


def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, results_dir):
    train_losses = []
    test_losses = []
    accuracies = []
    losses = []

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
            losses.append(loss.item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Calculate test loss and accuracy
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        test_loss = test_loss / len(val_loader.dataset)
        test_losses.append(test_loss)
        accuracy = 100 * correct / total
        accuracies.append(accuracy)

        # Save model to Results folder
        results.save_model_results(model, results_dir, epoch)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
        metrics_path = os.path.join(results_dir, 'epoch_metrics.txt')
        with open(metrics_path, 'a') as f:
            f.write(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n")

    # Save training results
    save_training_results(losses, train_losses, test_losses, num_epochs, accuracies, results_dir)


def main():
    # Initialize DataLoader
    train_loader, val_loader, eval_loader = get_dataloader(batch_size=32)

    # Set the number of classes based on your dataset
    num_classes = len(train_loader.dataset.classes) # Automatically set based on dataset
    model = get_resnet50_model(num_classes=num_classes, pretrained=True)

    # Check if GPU is available and move model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Print model summary
    print(model)

    # Calculate class weights
    train_labels = train_loader.dataset.targets
    classes = np.unique(train_labels) # Get unique class labels
    class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device) # Move to GPU

    # Plot data distribution
    # plot_data_distribution(train_loader, dataset_type="Training")
    # plot_total_images(train_loader, eval_loader, dataset_type="Food Dataset")

    # Define training parameters
    num_epochs = 15
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    base_results_dir = r"../Results"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(base_results_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Training model
    train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, results_dir)

    # Initialize OOD detection
    ood_loader = get_ood_loader(batch_size=32, num_samples=len(eval_loader.dataset))

    # temperature scaling
    class TemperatureScaledModel(torch.nn.Module):
        def __init__(self, model, temperature):
            super().__init__()
            self.model = model
            self.temperature = temperature

        def forward(self, inputs):
            outputs = self.model(inputs)
            return outputs / self.temperature

    # Temperature
    temperature = 1000
    scaled_model = TemperatureScaledModel(model, temperature)

    # Perturbation
    eps = 0.001  # Small perturbation magnitude for ODIN
    odin_detector = ODIN(scaled_model, eps=eps)

    # Evaluating ODIN
    print("Evaluating OOD detection with ODIN")
    odin_auroc = compute_odin_scores(scaled_model, eval_loader, ood_loader, odin_detector, device, results_dir)
    print(f"ODIN AUROC: {odin_auroc:.4f}")

    # Evaluate AUROC for MSP and MaxLog
    auroc_msp = OOD.compute_auroc(model, eval_loader, ood_loader, device, results_dir, method="MSP")
    auroc_maxlog = OOD.compute_auroc(model, eval_loader, ood_loader, device, results_dir, method="MaxLog")
    print(f"MSP AUROC: {auroc_msp:.4f}")
    print(f"MaxLog AUROC: {auroc_maxlog:.4f}")


if __name__ == '__main__':
    main()
