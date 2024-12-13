# Import Libraries
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

def get_resnet50_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    This function returns a modified ResNet50 model for the number of classes specified.
    
    Params:
    - num_classes (int): Number of output classes where default = 2.
    - pretrained (bool): Set to true (default) -> uses pretrained weights from ResNet50

    Returns:
    - nn.Module: Modified ResNet50 model with an updated fully connected layer.
    """
    # Load the pretrained ResNet50 model
    #model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Modify the fully connected layer to match the number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model
