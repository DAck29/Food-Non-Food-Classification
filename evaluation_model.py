import torch

def evaluate_model(model, dataloader, device):
    model.eval()
    corrects = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += (preds == targets.data).sum()
    print(f'accuracy: {100.0 * corrects / len(dataloader.dataset)}')