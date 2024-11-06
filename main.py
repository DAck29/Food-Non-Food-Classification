import random

from sklearn.utils import compute_class_weight

from data_loader import get_dataloader
from resnet_model import get_resnet50_model
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import results
from datetime import datetime
import os
import numpy as np

# Set random seed for reproducibility
seed = 42
random.seed(seed)  # Built-in Python random seed
np.random.seed(seed)  # Numpy random seed
torch.manual_seed(seed)  # PyTorch random seed

# Set seed for CUDA operations, if GPU is available
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def save_training_results(losses, dir_results):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(dir_results, exist_ok=True)
    results_path = os.path.join(dir_results, f"training_results_{timestamp}.txt")

    with open(results_path, "w") as f:
        f.write("Training Losses:\n")
        for i, loss in enumerate(losses):
            f.write(f"Iteration {i + 1}: Loss = {loss:.4f}\n")

    print(f"Training results saved to {results_path}")


def main():
    # Initialize DataLoader
    train_loader, val_loader, eval_loader = get_dataloader(batch_size=32, augmented=True)

    # Set the number of classes based on your dataset
    num_classes = len(train_loader.dataset.classes)  # Automatically set based on dataset
    model = get_resnet50_model(num_classes=num_classes, pretrained=True)

    # Check if GPU is available and move model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Print model summary
    print(model)

    # Calculate class weights
    train_labels = train_loader.dataset.targets  # Assuming your dataset has a 'targets' attribute
    classes = np.unique(train_labels)  # Get unique class labels
    class_weights = compute_class_weight('balanced', classes=classes, y=train_labels) # why balanced?
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)  # Move to GPU if necessary


    # Define training parameters
    num_epochs = 5
    losses = []
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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

            if (batch_idx + 1) % 50 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Plot training loss
    plt.figure(figsize=(12, 4))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Cross Entropy Loss')
    plt.show()

    # Save model to Results folder
    results_dir = results.save_model_results(model)
    # Save training results
    save_training_results(losses, results_dir)


if __name__ == '__main__':
    main()