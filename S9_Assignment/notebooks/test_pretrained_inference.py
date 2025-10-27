# Test Pretrained Model Inference
# This code tests if your data loader is working correctly by using a pretrained ResNet50

import torch
import torch.nn as nn
import torchvision.models as models
import json
from tqdm import tqdm

print("="*60)
print("TESTING PRETRAINED MODEL INFERENCE")
print("="*60)

# 1. Load pretrained ResNet50 from torchvision
print("\n1. Loading pretrained ResNet50...")
pretrained_model = models.resnet50(pretrained=True)
pretrained_model = pretrained_model.to(device)
pretrained_model.eval()
print("✓ Pretrained model loaded successfully")

# 2. Load ImageNet class names
print("\n2. Loading ImageNet class names...")
# You can download this file from: https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json
try:
    # Try to load from local file first
    with open('imagenet_class_labels.json', 'r') as f:
        class_names = json.load(f)
except:
    # Use a simple mapping if file not found
    print("⚠️  Class names file not found, using class indices instead")
    class_names = [f"class_{i}" for i in range(1000)]

# 3. Test on a single batch
print("\n3. Testing on a single batch...")
sample_batch = next(iter(val_loader))
images, labels = sample_batch[0].to(device), sample_batch[1].to(device)

with torch.no_grad():
    outputs = pretrained_model(images)
    probabilities = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    
    # Calculate accuracy
    correct = (predictions == labels).sum().item()
    batch_accuracy = 100.0 * correct / labels.size(0)
    
    print(f"Batch size: {images.shape[0]}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Batch accuracy: {batch_accuracy:.2f}% ({correct}/{labels.size(0)})")

# 4. Show detailed results for first 5 images
print("\n4. Detailed results for first 5 images:")
print("-" * 50)
for i in range(min(5, images.shape[0])):
    true_label = labels[i].item()
    pred_label = predictions[i].item()
    confidence = probabilities[i, pred_label].item()
    
    print(f"\nImage {i+1}:")
    print(f"  True label: {true_label} ({class_names[true_label] if true_label < len(class_names) else 'Unknown'})")
    print(f"  Predicted:  {pred_label} ({class_names[pred_label] if pred_label < len(class_names) else 'Unknown'})")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Correct:    {'✓' if true_label == pred_label else '✗'}")
    
    # Show top-5 predictions
    top5_probs, top5_indices = torch.topk(probabilities[i], 5)
    print("  Top-5 predictions:")
    for j, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
        class_name = class_names[idx.item()] if idx.item() < len(class_names) else f"class_{idx.item()}"
        print(f"    {j+1}. {class_name} ({prob.item():.2%})")

# 5. Test on multiple batches for better statistics
print("\n5. Testing on multiple batches...")
num_batches_to_test = 10
total_correct = 0
total_samples = 0
total_correct_top5 = 0

pretrained_model.eval()
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(val_loader):
        if batch_idx >= num_batches_to_test:
            break
            
        images, labels = images.to(device), labels.to(device)
        outputs = pretrained_model(images)
        
        # Top-1 accuracy
        _, predictions = outputs.max(1)
        correct = predictions.eq(labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)
        
        # Top-5 accuracy
        _, pred_top5 = outputs.topk(5, 1, largest=True, sorted=True)
        correct_top5 = pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5)).sum().item()
        total_correct_top5 += correct_top5
        
        print(f"  Batch {batch_idx+1}/{num_batches_to_test}: "
              f"Top-1: {100.*correct/labels.size(0):.1f}%, "
              f"Top-5: {100.*correct_top5/labels.size(0):.1f}%")

overall_acc = 100. * total_correct / total_samples
overall_acc_top5 = 100. * total_correct_top5 / total_samples

print(f"\nOverall statistics on {total_samples} samples:")
print(f"  Top-1 Accuracy: {overall_acc:.2f}%")
print(f"  Top-5 Accuracy: {overall_acc_top5:.2f}%")

# 6. Expected results
print("\n6. EXPECTED RESULTS:")
print("  • Pretrained ResNet50 should achieve ~76% top-1 accuracy on ImageNet val set")
print("  • Top-5 accuracy should be ~93%")
print("  • If accuracy is very low (<50%), there might be data loading issues")
print("  • If accuracy is 0% or 100%, labels might be misaligned")

# 7. Compare with your model
print("\n7. COMPARING WITH YOUR MODEL:")
print("Now let's test your model with pretrained=True...")

# Load your model with pretrained weights
your_model = resnet50(
    num_classes=1000,
    pretrained=True,  # Enable pretrained weights
    replace_maxpool_with_conv=config['replace_maxpool_with_conv']
)
your_model = your_model.to(device)
your_model.eval()

# Test your model on one batch
with torch.no_grad():
    outputs = your_model(images)
    _, predictions = outputs.max(1)
    correct = predictions.eq(labels).sum().item()
    your_acc = 100. * correct / labels.size(0)
    
print(f"Your model accuracy on last batch: {your_acc:.2f}%")
print(f"Torchvision model accuracy: {100.*correct/labels.size(0):.2f}%")

if abs(your_acc - 100.*correct/labels.size(0)) < 5:
    print("✓ Your model performs similarly to torchvision model!")
else:
    print("⚠️  There might be differences in model implementation or preprocessing")

# 8. Debugging data loader
print("\n8. DATA LOADER VERIFICATION:")
# Check if labels are in correct range
print(f"Label range in batch: {labels.min().item()} to {labels.max().item()}")
print(f"Expected range: 0 to 999")

# Check image statistics
print(f"Image shape: {images.shape}")
print(f"Image value range: [{images.min().item():.3f}, {images.max().item():.3f}]")
print("Expected normalized range: approximately [-2.5, 2.5]")

# Visualize predictions vs ground truth
print("\n9. VISUALIZATION CHECK:")
print("If running in notebook, uncomment the visualization code below:")
print("""
# Visualize some predictions
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.ravel()

for i in range(min(8, images.shape[0])):
    # Denormalize image
    img = images[i].cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    
    axes[i].imshow(img.permute(1, 2, 0))
    axes[i].set_title(f'True: {labels[i].item()}\\nPred: {predictions[i].item()}')
    axes[i].axis('off')

plt.suptitle('Pretrained Model Predictions')
plt.tight_layout()
plt.show()
""")

print("\n" + "="*60)
print("INFERENCE TEST COMPLETE")
print("="*60)