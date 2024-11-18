from sklearn.metrics import roc_auc_score, roc_curve
from resnet_model import get_resnet50_model
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


def compute_auroc(model, id_loader, ood_loader, device, results_dir, method="MSP"):
    
    # # -----------------------------------------------------------------------------
    # # use the already trained Model instead of training first
    # trained_model_path = '/storage/homefs/ma20e073/FoodClassifierScript/Results/2024-11-12_16-08-59/resnet50_model_20.pth'
    # num_classes = len(id_loader.dataset.classes)  # Automatically set based on dataset
    # model = get_resnet50_model(num_classes=num_classes)
    # checkpoint = torch.load(trained_model_path, map_location=torch.device('cpu'))  # Change 'cpu' to 'cuda' if using GPU
    # model.load_state_dict(checkpoint)
    # model.to(device)
    # # -----------------------------------------------------------------------------


    model.eval()
    id_scores = []
    ood_scores = []

    # Define score extraction function
    if method == "MSP":
        score_fn = lambda outputs: torch.softmax(outputs, dim=1).max(dim=1).values # Max Probability from softmax output of Model 
    elif method == "MaxLog":
        score_fn = lambda outputs: outputs.max(dim=1).values # Max raw output before softmax
    else:
        raise ValueError("Invalid method. Choose 'MSP' or 'MaxLog'.")

    # Process ID data
    with torch.no_grad():
        for inputs, _ in id_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            id_scores.extend(score_fn(outputs).cpu().numpy())

    # Process OOD data
    with torch.no_grad():
        for inputs, _ in ood_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            ood_scores.extend(score_fn(outputs).cpu().numpy())

    # Combine scores and labels
    scores = np.array(id_scores + ood_scores)
    labels = np.array([1] * len(id_scores) + [0] * len(ood_scores))

    # Calculate AUROC
    auroc = roc_auc_score(labels, scores)

    # Calculate TPR and FPR for ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)

    # Plot AUROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{method} AUROC: {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')  # Random guess line
    plt.xlabel('False Positive (In-Distribution) Rate')
    plt.ylabel('True Positive (In-Distribution) Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve ({method})')
    plt.legend()
    plt.grid()

    # Save plot in the current folder
    auroc_path = os.path.join(results_dir, f"auroc_{method.lower()}_plot.png")
    plt.savefig(auroc_path)
    print(f"AUROC plot saved to {auroc_path}")
    plt.close()


    return auroc
