import torch
import os
import torch.nn as nn
from datetime import datetime

def save_model_results(model: nn.Module, base_results_dir: str = "Results") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(base_results_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, f"resnet50_model_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return results_dir
