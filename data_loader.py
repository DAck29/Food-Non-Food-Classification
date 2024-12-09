import os
import random
import os
import random
import torchvision
from torch.utils.data import DataLoader, Subset
from typing import Tuple

# def get_dataloader(batch_size: int, dataset_dir: str = r"C:\Users\manu_\OneDrive - Universitaet Bern\03 HS24 UniBe-VIVO\05 Diabetes Management\GitHub_Clone\Dataset\food_data") -> Tuple[
def get_dataloader(batch_size: int, dataset_dir: str = "/storage/homefs/da17u029/DD_DM/Dataset/food_data") -> Tuple[
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    evaluation_loader = DataLoader(evaluation_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, validation_loader, evaluation_loader





def get_ood_loader(batch_size: int, dataset_dir: str = "/storage/homefs/da17u029/DD_DM/Dataset/cifar-10-batches-py", num_samples: int = 10000, seed: int = 42) -> DataLoader:

    # Define transformation for the OOD dataset (CIFAR-10)
    transform_ood = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),  # Resize to match the input size for ResNet
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load CIFAR-10 dataset (testing set is used as OOD data)
    ood_dataset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=transform_ood)

    random.seed(seed) # fix random seed

    # sample num_samples indices from the OOD dataset
    all_indices = list(range(len(ood_dataset)))
    selected_indices = random.sample(all_indices, num_samples)

    # Create a subset with num_samples images
    ood_subset = Subset(ood_dataset, selected_indices)

    ood_loader = DataLoader(ood_subset, batch_size=batch_size, shuffle=False, num_workers=1)

    return ood_loader