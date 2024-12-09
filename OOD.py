from sklearn.metrics import roc_auc_score, roc_curve
from resnet_model import get_resnet50_model
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from pytorch_ood.detector import ODIN


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
        score_fn = lambda outputs: torch.softmax(outputs, dim=1).max(
            dim=1).values  # Max Probability from softmax output of Model
    elif method == "MaxLog":
        score_fn = lambda outputs: outputs.max(dim=1).values  # Max raw output before softmax
    elif method == "ODIN":
        odin_detector = ODIN(model, temperature=1000, eps=0.05)
        score_fn = lambda inputs: odin_detector(inputs)  # Using temperature scaling for ODIN
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

    # Plot AUROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'{method} AUROC: {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')  # Random guess line
    plt.xlabel('False Positive (In-Distribution) Rate', fontsize=14)
    plt.ylabel('True Positive (In-Distribution) Rate', fontsize=14)
    plt.title(f'Receiver Operating Characteristic (ROC) Curve ({method})', fontsize=16)
    plt.legend()
    plt.grid()

    # Save plot in the current folder
    auroc_path = os.path.join(results_dir, f"auroc_{method.lower()}_plot.png")
    plt.savefig(auroc_path)
    print(f"AUROC plot saved to {auroc_path}")
    plt.close()

    print(f"FPR at 95% TPR: {fpr_at_95_tpr:.4f}")

    return auroc, fpr_at_95_tpr


def compute_auroc_epoch(model, id_loader, ood_loader, device, results_dir, method="MSP", epoch_nr=None):
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

    # Plot AUROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'{method} AUROC: {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
    plt.xlabel('False Positive (In-Distribution) Rate', fontsize=14)
    plt.ylabel('True Positive (In-Distribution) Rate', fontsize=14)
    plt.title(f'Receiver Operating Characteristic (ROC) Curve ({method})', fontsize=16)
    plt.legend()
    plt.grid()

    # Save plot with epoch number in filename if provided
    auroc_filename = f"auroc_{method.lower()}_epoch_{epoch_nr}_plot.png" if epoch_nr is not None else f"auroc_{method.lower()}_plot.png"
    auroc_path = os.path.join(results_dir, auroc_filename)
    plt.savefig(auroc_path)
    print(f"AUROC plot saved to {auroc_path}")
    plt.close()

    print(f"FPR at 95% TPR: {fpr_at_95_tpr:.4f}")

    return auroc, fpr_at_95_tpr


def plot_auroc_curves(roc_data_dict, results_dir):
    # Plot all AUROC curves on the same plot
    plt.figure()
    for method, (fpr, tpr, auroc) in roc_data_dict.items():
        plt.plot(fpr, tpr, label=f'{method} AUROC: {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')  # Random guess line
    plt.xlabel('False Positive (In-Distribution) Rate', fontsize=14)
    plt.ylabel('True Positive (In-Distribution) Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curves for Different Methods', fontsize=16)
    plt.legend()
    plt.grid()

    # Save plot in the current folder
    auroc_path = os.path.join(results_dir, 'combined_auroc_plot.png')
    plt.savefig(auroc_path)
    print(f"Combined AUROC plot saved to {auroc_path}")
    plt.close()


def plot_auroc_curves_epoch(roc_data_dict, results_dir, epoch_nr=None):
    # Plot all AUROC curves on the same plot
    plt.figure()
    for method, (fpr, tpr, auroc) in roc_data_dict.items():
        plt.plot(fpr, tpr, label=f'{method} AUROC: {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
    plt.xlabel('False Positive (In-Distribution) Rate', fontsize=14)
    plt.ylabel('True Positive (In-Distribution) Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curves for Different Methods', fontsize=16)
    plt.legend()
    plt.grid()

    # Save plot with epoch number in filename if provided
    auroc_filename = f"combined_auroc_epoch_{epoch_nr}.png" if epoch_nr is not None else 'combined_auroc_plot.png'
    auroc_path = os.path.join(results_dir, auroc_filename)
    plt.savefig(auroc_path)
    print(f"Combined AUROC plot saved to {auroc_path}")
    plt.close()


def plot_metrics_bar_chart(auroc_dict, fpr_at_95_dict, results_dir):
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

    auroc_bar_path = os.path.join(results_dir, 'auroc_bar_chart.png')
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

    fpr_bar_path = os.path.join(results_dir, 'fpr_at_95_tpr_bar_chart.png')
    plt.savefig(fpr_bar_path)
    print(f"FPR at 95% TPR bar chart saved to {fpr_bar_path}")
    plt.close()


def plot_metrics_bar_chart_epoch(auroc_dict, fpr_at_95_dict, results_dir, epoch_nr=None):
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
