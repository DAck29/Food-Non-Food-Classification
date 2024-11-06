import os
import torchvision
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm

def get_dataloader(batch_size: int, dataset_dir: str = "../Dataset/food_data") -> Tuple[
    DataLoader, DataLoader, DataLoader]:

    transform_augmented = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),  # Fix image size
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.RandomRotation(20),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ColorJitter(0.1, 0.1, 0.1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
            torchvision.transforms.RandomErasing()
        ])

    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),  # Fix image size
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
        ])

    # Load dataset and apply transformation
    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(dataset_dir, "training"), transform=transform_augmented)
    validation_dataset = torchvision.datasets.ImageFolder(root=os.path.join(dataset_dir, "validation"),
                                                          transform=transform)
    evaluation_dataset = torchvision.datasets.ImageFolder(root=os.path.join(dataset_dir, "evaluation"),
                                                          transform=transform)

    # Initialize dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    evaluation_loader = DataLoader(evaluation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, validation_loader, evaluation_loader
