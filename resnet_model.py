import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights



def get_resnet50_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:

    # Load the pretrained ResNet50 model
    #model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Modify the fully connected layer to match the number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model