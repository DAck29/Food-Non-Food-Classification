import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

class ImageDataset(torch.utils.data.Dataset):
    """
    IMAGEDATASET create custom dataset class
    """
    def __init__(self, directory, transform=None):
        """
        :param directory:
        :param transform:
        """
        self.data_dir = directory
        self.images = []
        self.labels = []
        self.classes = sorted(os.listdir(directory))

        # Access every subfolder
        for idx, class_dir in enumerate(self.classes):
            class_path = os.path.join(directory, class_dir)
            if os.path.isdir(class_path):

                # Access every image of a subfolder and attach a label to it
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    if os.path.isfile(img_path):
                        self.images.append(img_path)
                        self.labels.append(idx)

        # Apply transformation to normalize the image
        self.transform = transform

    def __len__(self):
        """
        :return: Returns the length of the dataset
        """
        return len(self.images)

    # Defining the method th get an image from the dataset
    def __getitem__(self, index):
        """
        :param index:
        :return: Return the image and the label of the dataset based on the subfolder
        """
        image_path = self.images[index]
        image = Image.open(image_path)

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        label = self.labels[index]
        return image, label


def own_data_logger():
    """
    :return:
    """
    data_path = 'Dataset/food_data/training'
    dataset = ImageDataset(data_path)
    dataset_length = len(dataset)
    print('Number of training examples:', dataset_length)
    # Zufälligen Index auswählen und Bild anzeigen
    random_index = np.random.randint(0, dataset_length - 1)
    image, label = dataset[random_index]  # Bild und Label separat extrahieren

    # Bild anzeigen
    plt.imshow(image)
    plt.title(f"Class: {dataset.classes[label]}")
    plt.show()

def load_data_dataloader(batch: int):
    # Transformation: Bilder in Tensors konvertieren und normalisieren
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),  # Bildgröße anpassen
        torchvision.transforms.ToTensor(),  # Bild in einen PyTorch Tensor umwandeln
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Optional: Normalisierung
    ])

    # Dataset laden und Transformation anwenden
    train_dataset = torchvision.datasets.ImageFolder(root="Dataset/food_data/training", transform=transform)
    validation_dataset = torchvision.datasets.ImageFolder(root="Dataset/food_data/validation", transform=transform)
    evaluation_dataset = torchvision.datasets.ImageFolder(root="Dataset/food_data/evaluation", transform=transform)

    # DataLoader initialisieren
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=batch, shuffle=True, num_workers=4)
    evaluation_loader = DataLoader(evaluation_dataset, batch_size=batch, shuffle=True, num_workers=4)

    # Example of how to iterate through the data
    for images, labels in train_loader:
        print(f'Batch of images has shape: {images.shape}')
        print(f'Batch of labels has shape: {labels.shape}')