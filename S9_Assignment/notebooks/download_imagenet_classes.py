# Download ImageNet class names

import json
import urllib.request

print("Downloading ImageNet class names...")

# Option 1: Simple labels (recommended)
try:
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    with urllib.request.urlopen(url) as response:
        class_names = json.loads(response.read())
    
    # Save to file
    with open('imagenet_class_labels.json', 'w') as f:
        json.dump(class_names, f)
    
    print(f"✓ Downloaded {len(class_names)} class names")
    print(f"✓ Saved to imagenet_class_labels.json")
    
    # Show first 10 classes
    print("\nFirst 10 classes:")
    for i in range(10):
        print(f"  {i}: {class_names[i]}")
        
except Exception as e:
    print(f"Error downloading: {e}")

# Option 2: Create a mapping manually (backup)
print("\nAlternatively, here's code to create a basic mapping:")
print("""
# Basic ImageNet class mapping (first 10 classes)
imagenet_classes = {
    0: "tench",
    1: "goldfish", 
    2: "great white shark",
    3: "tiger shark",
    4: "hammerhead",
    5: "electric ray",
    6: "stingray",
    7: "cock",
    8: "hen",
    9: "ostrich",
    # ... continues to 999
}
""")

# Option 3: Download full mapping with synsets
print("\nFor full mapping with WordNet IDs:")
print("""
# Download the full ImageNet class index
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
with urllib.request.urlopen(url) as response:
    class_index = json.loads(response.read())

# Convert to simple list
class_names = [class_index[str(i)][1] for i in range(1000)]
""")