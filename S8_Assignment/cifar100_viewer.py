#!/usr/bin/env python3
"""
Simple CIFAR-100 Dataset Viewer
Displays images from CIFAR-100 binary files with their labels.
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def unpickle(file):
    """Load a CIFAR-100 pickled file."""
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def display_images(data_file, meta_file=None, count=9):
    """Display sample images from CIFAR-100 dataset."""
    
    # Load data
    print(f"Loading: {data_file}")
    data = unpickle(data_file)
    
    # Extract images and labels
    images = data[b'data']
    
    # Check for CIFAR-100 specific labels
    if b'fine_labels' in data:
        labels = data[b'fine_labels']
        print("Detected CIFAR-100 format (fine labels)")
    else:
        labels = data[b'labels']
        print("Detected CIFAR-10 format")
    
    # Load label names if meta file provided
    label_names = None
    if meta_file and Path(meta_file).exists():
        print(f"Loading labels from: {meta_file}")
        meta = unpickle(meta_file)
        if b'fine_label_names' in meta:
            label_names = [name.decode('utf-8') for name in meta[b'fine_label_names']]
        elif b'label_names' in meta:
            label_names = [name.decode('utf-8') for name in meta[b'label_names']]
    
    print(f"Total images: {len(images)}")
    print(f"Displaying: {count} images\n")
    
    # Setup plot
    rows = int(np.sqrt(count))
    cols = int(np.ceil(count / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if count == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Display each image
    for i in range(count):
        # Reshape image from flat array to 32x32x3
        img_flat = images[i]
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:, :, 0] = img_flat[:1024].reshape(32, 32)      # Red channel
        img[:, :, 1] = img_flat[1024:2048].reshape(32, 32)  # Green channel
        img[:, :, 2] = img_flat[2048:3072].reshape(32, 32)  # Blue channel
        
        # Show image
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # Set title
        label_id = labels[i]
        if label_names:
            title = f"{label_names[label_id]}\n(ID: {label_id})"
        else:
            title = f"Label: {label_id}"
        axes[i].set_title(title)
    
    # Hide unused subplots
    for i in range(count, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"CIFAR-100: {Path(data_file).name}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cifar100_viewer.py <data_file> [meta_file] [num_images]")
        print("\nExamples:")
        print("  python cifar100_viewer.py train")
        print("  python cifar100_viewer.py test meta")
        print("  python cifar100_viewer.py train meta 16")
        sys.exit(1)
    
    data_file = sys.argv[1]
    meta_file = sys.argv[2] if len(sys.argv) > 2 else None
    num_images = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    
    if not Path(data_file).exists():
        print(f"Error: File '{data_file}' not found")
        sys.exit(1)
    
    display_images(data_file, meta_file, num_images)