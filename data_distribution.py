import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

def plot_data_distribution(data_loader: DataLoader, dataset_type: str = "Dataset"):
  
    # Get class labels and counts
    labels = data_loader.dataset.targets
    classes, counts = np.unique(labels, return_counts=True)
    
    # Retrieve class names directly from the dataset
    class_names = data_loader.dataset.classes  
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, counts, color='skyblue') 
    plt.xlabel("Class Labels")
    plt.ylabel("Count")
    plt.title(f"{dataset_type} Data Distribution by Class")
    plt.xticks(rotation=45, ha='right')

    # Add counts above each bar
    for i, count in enumerate(counts):
        plt.text(i, count + 5, str(count), ha='center', va='bottom')

    # Save and display
    plt.tight_layout()  # Adjust layout to accommodate rotated labels
    plt.savefig(f"{dataset_type.lower()}_data_distribution_histogram.png", format='png')



def plot_total_images(train_loader: DataLoader, eval_loader: DataLoader, dataset_type: str = "Dataset"):
 
    # Calculate total number of images in each dataset
    total_train_images = len(train_loader.dataset)
    total_eval_images = len(eval_loader.dataset)

    # Data for plotting
    categories = ['Training', 'Evaluation']
    image_counts = [total_train_images, total_eval_images]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.bar(categories, image_counts, color=['blue', 'orange'])
    plt.xlabel('Dataset Type')
    plt.ylabel('Total Number of Images')
    plt.title(f'Total Number of Images in {dataset_type}')
    plt.xticks(rotation=0)
    
    # Annotate counts on each bar
    for i, count in enumerate(image_counts):
        plt.text(i, count + 10, str(count), ha='center', va='bottom')

    # Save and display
    plt.tight_layout()
    plt.savefig(f"{dataset_type.lower()}_total_images_barplot.png", format='png')
