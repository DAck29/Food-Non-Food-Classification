import torch
import os
import torch.nn as nn
from datetime import datetime

def save_model_results(model: nn.Module, results_dir: str, epoch_nr: int) -> None:
    model_path = os.path.join(results_dir, f"resnet50_model_{epoch_nr+1}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")