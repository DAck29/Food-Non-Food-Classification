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




def plot_combined_data_distribution(
    eval_loader: DataLoader,
    ood_loader: DataLoader,
    dataset_type_eval: str = "Evaluation Dataset (ID)",
    dataset_type_ood: str = "OOD Dataset"
):

    # Evaluation Dataset (ID)
    eval_labels = eval_loader.dataset.targets
    eval_classes, eval_counts = np.unique(eval_labels, return_counts=True)
    eval_class_names = eval_loader.dataset.classes  # Class names

    # OOD Dataset
    ood_labels = [ood_loader.dataset.dataset.targets[idx] for idx in ood_loader.dataset.indices]
    ood_classes, ood_counts = np.unique(ood_labels, return_counts=True)
    ood_class_names = ood_loader.dataset.dataset.classes

    # Combine class names for proper alignment
    class_names_combined = sorted(set(eval_class_names).union(ood_class_names))

    # map class names to corresponding count in evaluation set
    eval_class_dict = dict(zip(eval_class_names, eval_counts))

    # map class names to corresponding count in OOD set
    ood_class_dict = dict(zip(ood_class_names, ood_counts))

    # Now align counts based on the combined class names
    eval_counts_aligned = [eval_class_dict.get(cls, 0) for cls in class_names_combined]
    ood_counts_aligned = [ood_class_dict.get(cls, 0) for cls in class_names_combined]

    print("Aligned Evaluation Counts:", eval_counts_aligned)
    print("Aligned OOD Counts:", ood_counts_aligned)

    # Total number of images
    total_eval_images = sum(eval_counts_aligned)
    total_ood_images = sum(ood_counts_aligned)

    # Plotting
    x = np.arange(len(class_names_combined))  # Bar positions
    width = 0.4  # Bar width 

    plt.figure(figsize=(20, 12))

    # Plot bars
    plt.bar(x - width / 2, eval_counts_aligned, width, label=dataset_type_eval, color='skyblue')
    plt.bar(x + width / 2, ood_counts_aligned, width, label=dataset_type_ood, color='lightcoral')

    plt.xlabel("Class Labels", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title("Data Distribution: ID vs. OOD", fontsize=16)
    plt.xticks(x, class_names_combined, rotation=45, ha='center', fontsize=12)

    plt.legend(fontsize=14)

    # Annotate bar counts with smaller font size
    for i, (eval_count, ood_count) in enumerate(zip(eval_counts_aligned, ood_counts_aligned)):
        # Only annotate if the count is greater than 0
        if eval_count > 0:
            plt.text(i - width / 2, eval_count + 2, str(eval_count), ha='center', va='bottom', fontsize=10)
        if ood_count > 0:
            plt.text(i + width / 2, ood_count + 2, str(ood_count), ha='center', va='bottom', fontsize=10)

    # Annotate total images for ID and OOD datasets
    plt.text(0.98, 0.88, f"Total ID: {total_eval_images}", transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=14, color='skyblue', fontweight='bold')
    plt.text(0.98, 0.84, f"Total OOD: {total_ood_images}", transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=14, color='lightcoral', fontweight='bold')

    plt.ylim(bottom=0)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.25)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig("ID_OOD_combined_data_distribution.png", format='png')
    plt.show()



def plot_total_images(train_loader: DataLoader, val_loader: DataLoader, eval_loader: DataLoader, dataset_type: str = "Dataset"):
 
    # Calculate total number of images in each dataset
    total_train_images = len(train_loader.dataset)
    total_val_images = len(val_loader.dataset)
    total_val_images = len(val_loader.dataset)
    total_eval_images = len(eval_loader.dataset)


    # Data for plotting
    categories = ['Training', 'Validation', 'Evaluation']
    image_counts = [total_train_images, total_val_images, total_eval_images]

    # Plotting
    plt.figure(figsize=(8, 4))
    plt.bar(categories, image_counts, color=['skyblue', 'plum', 'darkorange'])
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
