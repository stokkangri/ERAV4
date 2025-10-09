#!/usr/bin/env python3
"""
CIFAR-100 Dataset Visualizer
Reads CIFAR-100 binary files and displays sample images with their labels.
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def unpickle(file):
    """
    Unpickle a CIFAR-100 binary file.
    
    Args:
        file: Path to the binary file
        
    Returns:
        Dictionary containing the data
    """
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data


def reshape_image(flat_image):
    """
    Reshape a flat CIFAR-100 image array into a 32x32x3 image.
    
    Args:
        flat_image: 1D array of 3072 values (32*32*3)
        
    Returns:
        3D numpy array of shape (32, 32, 3)
    """
    # The image is stored as [R_channel, G_channel, B_channel]
    # Each channel is 32x32 = 1024 values
    red = flat_image[:1024].reshape(32, 32)
    green = flat_image[1024:2048].reshape(32, 32)
    blue = flat_image[2048:3072].reshape(32, 32)
    
    # Stack the channels to create RGB image
    image = np.stack([red, green, blue], axis=2)
    return image


def load_label_names(meta_file):
    """
    Load the label names from the meta file.
    
    Args:
        meta_file: Path to the meta binary file
        
    Returns:
        Dictionary with label names
    """
    try:
        meta_data = unpickle(meta_file)
        
        # The keys might be bytes, so we need to handle both cases
        if b'fine_label_names' in meta_data:
            # CIFAR-100 has fine and coarse labels
            fine_labels = [label.decode('utf-8') if isinstance(label, bytes) else label 
                          for label in meta_data[b'fine_label_names']]
            coarse_labels = [label.decode('utf-8') if isinstance(label, bytes) else label 
                            for label in meta_data[b'coarse_label_names']]
            return {'fine': fine_labels, 'coarse': coarse_labels}
        elif b'label_names' in meta_data:
            # Fallback for CIFAR-10 style
            labels = [label.decode('utf-8') if isinstance(label, bytes) else label 
                     for label in meta_data[b'label_names']]
            return {'labels': labels}
        else:
            print("Warning: Could not find label names in meta file")
            return None
    except Exception as e:
        print(f"Warning: Could not load meta file: {e}")
        return None


def visualize_images(data_file, meta_file=None, num_images=16):
    """
    Visualize sample images from a CIFAR-100 data file.
    
    Args:
        data_file: Path to the data binary file (train or test)
        meta_file: Path to the meta binary file (optional)
        num_images: Number of images to display
    """
    # Load the data
    print(f"Loading data from: {data_file}")
    data_dict = unpickle(data_file)
    
    # Extract data and labels
    # Keys might be bytes, so handle both cases
    if b'data' in data_dict:
        images = data_dict[b'data']
        if b'fine_labels' in data_dict:
            # CIFAR-100 specific
            fine_labels = data_dict[b'fine_labels']
            coarse_labels = data_dict.get(b'coarse_labels', None)
        else:
            # CIFAR-10 style fallback
            fine_labels = data_dict[b'labels']
            coarse_labels = None
    else:
        print("Error: Could not find 'data' key in the file")
        return
    
    # Load label names if meta file is provided
    label_names = None
    if meta_file and Path(meta_file).exists():
        label_names = load_label_names(meta_file)
    
    # Print dataset info
    print(f"Dataset shape: {images.shape}")
    print(f"Number of images: {len(images)}")
    print(f"Number of labels: {len(fine_labels)}")
    if coarse_labels:
        print(f"Number of coarse labels: {len(coarse_labels)}")
    
    # Calculate grid size for visualization
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(f'CIFAR-100 Sample Images from {Path(data_file).name}', fontsize=16)
    
    # Flatten axes array for easier indexing
    axes = axes.flatten()
    
    # Display images
    for i in range(min(num_images, len(images))):
        # Get and reshape the image
        image = reshape_image(images[i])
        
        # Display the image
        axes[i].imshow(image)
        axes[i].axis('off')
        
        # Create title with label
        if label_names:
            if 'fine' in label_names:
                # CIFAR-100 with fine and coarse labels
                fine_name = label_names['fine'][fine_labels[i]]
                title = f"Fine: {fine_name}\n(ID: {fine_labels[i]})"
                if coarse_labels and 'coarse' in label_names:
                    coarse_name = label_names['coarse'][coarse_labels[i]]
                    title += f"\nCoarse: {coarse_name}"
            else:
                # CIFAR-10 style
                label_name = label_names['labels'][fine_labels[i]]
                title = f"{label_name}\n(ID: {fine_labels[i]})"
        else:
            title = f"Label ID: {fine_labels[i]}"
            if coarse_labels:
                title += f"\nCoarse ID: {coarse_labels[i]}"
        
        axes[i].set_title(title, fontsize=8)
    
    # Hide any unused subplots
    for i in range(min(num_images, len(images)), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print("\nLabel distribution (first 100 images):")
    unique_labels, counts = np.unique(fine_labels[:100], return_counts=True)
    for label, count in zip(unique_labels[:10], counts[:10]):  # Show first 10 unique labels
        if label_names and 'fine' in label_names:
            label_name = label_names['fine'][label]
            print(f"  {label_name} (ID {label}): {count} images")
        else:
            print(f"  Label {label}: {count} images")


def main():
    """Main function to handle command line arguments and run visualization."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_cifar100.py <data_file> [meta_file] [num_images]")
        print("\nExample:")
        print("  python visualize_cifar100.py train meta")
        print("  python visualize_cifar100.py test meta 25")
        print("  python visualize_cifar100.py /path/to/train /path/to/meta")
        sys.exit(1)
    
    data_file = sys.argv[1]
    meta_file = sys.argv[2] if len(sys.argv) > 2 else None
    num_images = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    
    # Check if file exists
    if not Path(data_file).exists():
        print(f"Error: Data file '{data_file}' not found")
        sys.exit(1)
    
    if meta_file and not Path(meta_file).exists():
        print(f"Warning: Meta file '{meta_file}' not found, proceeding without label names")
        meta_file = None
    
    # Visualize the images
    try:
        visualize_images(data_file, meta_file, num_images)
    except Exception as e:
        print(f"Error processing files: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()