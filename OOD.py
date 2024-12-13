from sklearn.metrics import roc_auc_score, roc_curve
import torch
#import os
import numpy as np
#import matplotlib.pyplot as plt
from pytorch_ood.detector import ODIN

import results
#from results import plot_auroc_curve


def compute_auroc_fpr95(model, id_loader, ood_loader, device, results_dir, method="MSP", epoch_nr=None):
    """
        This function computes the AUROC and FPR95 score on the test and OOD set.

        Args:
            model: Trained multi-food classification model.
            id_loader: in-distribution loader (Multi-food classification).
            ood_loader: out-of-distribution loader (OOD).
            device: device to use.
            results_dir: results directory.
            method: method for computing AUROC.
            epoch_nr: number of epochs.

        Returns:
            None
    """
    model.eval()
    id_scores = []
    ood_scores = []

    # Define score extraction function
    if method == "MSP":
        score_fn = lambda outputs: torch.softmax(outputs, dim=1).max(
            dim=1).values  # Max Probability from softmax output of Model
    elif method == "MaxLog":
        score_fn = lambda outputs: outputs.max(dim=1).values  # Max raw output before softmax
    elif method == "ODIN":
        odin_detector = ODIN(model, temperature=1000, eps=0.05)
        score_fn = lambda inputs: -odin_detector(inputs)  # Using temperature scaling for ODIN
    else:
        raise ValueError("Invalid method. Choose 'MSP' or 'MaxLog'.")

    # Process ID data
    with torch.no_grad():
        for inputs, _ in id_loader:
            inputs = inputs.to(device)
            if method == "ODIN":
                id_scores.extend(score_fn(inputs).cpu().numpy())  # ODIN method takes inputs directly
            else:
                outputs = model(inputs)
                id_scores.extend(score_fn(outputs).cpu().numpy())

    # Process OOD data
    with torch.no_grad():
        for inputs, _ in ood_loader:
            inputs = inputs.to(device)
            if method == "ODIN":
                ood_scores.extend(score_fn(inputs).cpu().numpy())  # ODIN method takes inputs directly
            else:
                outputs = model(inputs)
                ood_scores.extend(score_fn(outputs).cpu().numpy())

    # Combine scores and labels
    scores = np.array(id_scores + ood_scores)
    labels = np.array([1] * len(id_scores) + [0] * len(ood_scores))

    # Calculate AUROC
    auroc = roc_auc_score(labels, scores)

    # Calculate TPR and FPR for ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)

    # Calculate FPR at 95% TPR
    idx_95_tpr = np.where(tpr >= 0.95)[0][0]
    fpr_at_95_tpr = fpr[idx_95_tpr]

    # Plot ROC curve
    results.plot_auroc_curve(fpr, tpr, auroc, method, results_dir, epoch_nr)
    print(f"FPR at 95% TPR: {fpr_at_95_tpr:.4f}")

    return auroc, fpr_at_95_tpr

