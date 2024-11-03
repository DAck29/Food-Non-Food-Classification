from data_loader import get_dataloader
from resnet_model import get_resnet50_model
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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

    # Define training parameters
    num_epochs = 5
    losses = []
    criterion = nn.CrossEntropyLoss()
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


if __name__ == '__main__':
    main()