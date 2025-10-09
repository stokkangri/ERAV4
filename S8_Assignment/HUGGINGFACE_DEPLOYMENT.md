# 🤗 Hugging Face Deployment Guide

This guide will walk you through deploying your trained ResNet-50 CIFAR-100 classifier to Hugging Face Spaces.

## Prerequisites

1. **Trained Model**: Ensure you have `best_model_binary.pth` from your training
2. **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co)
3. **Git**: Install Git on your system

## Step 1: Create a New Space on Hugging Face

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in the details:
   - **Space name**: `cifar100-resnet50-classifier` (or your preferred name)
   - **License**: Choose appropriate license (e.g., MIT)
   - **Select SDK**: Choose **Gradio**
   - **SDK Version**: Leave as default
   - **Space hardware**: **CPU Basic** (free tier) or **GPU** if you have access
   - **Visibility**: Public or Private as preferred
4. Click **"Create Space"**

## Step 2: Clone Your Space Locally

```bash
# Clone your space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/cifar100-resnet50-classifier
cd cifar100-resnet50-classifier
```

## Step 3: Prepare Files for Upload

### Required Files Structure:
```
cifar100-resnet50-classifier/
├── app.py                    # Main Gradio application
├── requirements.txt          # Python dependencies
├── best_model_binary.pth    # Your trained model weights
├── README.md                 # Space documentation
└── examples/                 # Optional: Example images
    ├── cat.jpg
    ├── car.jpg
    └── ...
```

### Copy Files:
```bash
# Copy the application file
cp /path/to/your/app.py .

# Copy requirements
cp /path/to/your/requirements.txt .

# Copy your trained model
cp /path/to/your/best_model_binary.pth .
```

## Step 4: Create Space README

Create a `README.md` file in your Space directory:

```markdown
---
title: CIFAR-100 ResNet-50 Classifier
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# CIFAR-100 Image Classifier with ResNet-50

This Space hosts a ResNet-50 model trained from scratch on CIFAR-100 dataset.

## Model Details
- **Architecture**: ResNet-50 (adapted for 32x32 images)
- **Dataset**: CIFAR-100 (100 classes)
- **Parameters**: 23.7M
- **Training**: OneCycle learning rate policy

## Usage
Upload an image to get predictions for 100 different categories including animals, vehicles, household items, and more!

## Performance
- Top-1 Accuracy: ~75-78%
- Top-5 Accuracy: ~93-95%
```

## Step 5: Add Example Images (Optional)

Create an `examples` folder and add sample images:

```bash
mkdir examples
# Add some 32x32 or small images for testing
# You can use CIFAR-100 test images or any small images
```

## Step 6: Test Locally (Optional)

Before uploading, test your app locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Visit `http://localhost:7860` to test the interface.

## Step 7: Upload to Hugging Face

### Using Git:

```bash
# Add all files
git add .

# Commit
git commit -m "Initial deployment of CIFAR-100 ResNet-50 classifier"

# Push to Hugging Face
git push
```

### Using Git LFS for Large Files:

If your model file is large (>10MB), use Git LFS:

```bash
# Install git-lfs if not already installed
git lfs install

# Track the model file
git lfs track "*.pth"
git add .gitattributes

# Add and commit
git add best_model_binary.pth
git commit -m "Add model weights"

# Push
git push
```

## Step 8: Monitor Deployment

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/cifar100-resnet50-classifier`
2. Watch the **"Building"** status in the top right
3. Once built, your app will be live!

## Alternative: Direct Upload via Web Interface

If you prefer not to use Git:

1. Go to your Space on Hugging Face
2. Click **"Files and versions"** tab
3. Click **"Add file"** → **"Upload files"**
4. Drag and drop or select:
   - `app.py`
   - `requirements.txt`
   - `best_model_binary.pth`
5. Add commit message and click **"Commit changes to main"**

## Troubleshooting

### Common Issues:

1. **"Module not found" error**:
   - Check `requirements.txt` includes all dependencies
   - Ensure correct versions are specified

2. **"Model file not found" error**:
   - Verify `best_model_binary.pth` is uploaded
   - Check the filename matches in `app.py`

3. **"Out of memory" error**:
   - Consider using CPU Basic tier instead of GPU
   - Reduce batch size if processing multiple images

4. **Build fails**:
   - Check logs in the Space's **"Logs"** tab
   - Ensure `app.py` has correct syntax
   - Verify Gradio version compatibility

### Checking Logs:

```bash
# View build logs
# Go to your Space → Settings → Logs

# Or use Hugging Face CLI
pip install huggingface-hub
huggingface-cli repo logs YOUR_USERNAME/cifar100-resnet50-classifier
```

## Step 9: Share Your Space

Once deployed, you can:

1. **Get the public URL**: `https://huggingface.co/spaces/YOUR_USERNAME/cifar100-resnet50-classifier`
2. **Embed in websites**: Click "Embed this Space" for iframe code
3. **Get API endpoint**: Use the API tab for programmatic access

### Example API Usage:

```python
import requests

# Your Space API endpoint
API_URL = "https://YOUR_USERNAME-cifar100-resnet50-classifier.hf.space/api/predict"

# Send image for prediction
with open("test_image.jpg", "rb") as f:
    response = requests.post(API_URL, files={"file": f})
    
print(response.json())
```

## Optional Enhancements

### 1. Add Model Card

Create `model_card.md`:

```markdown
# Model Card: ResNet-50 CIFAR-100

## Model Details
- Developed by: [Your Name]
- Model type: Convolutional Neural Network
- Architecture: ResNet-50
- Task: Image Classification

## Training Data
- Dataset: CIFAR-100
- Size: 50,000 training images, 10,000 test images
- Classes: 100 fine-grained categories

## Performance
| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 77.5% |
| Top-5 Accuracy | 94.2% |
| Parameters | 23.7M |
```

### 2. Add Configuration File

Create `config.json`:

```json
{
  "model_name": "ResNet-50 CIFAR-100",
  "num_classes": 100,
  "input_size": [32, 32],
  "normalize_mean": [0.5071, 0.4867, 0.4408],
  "normalize_std": [0.2675, 0.2565, 0.2761],
  "architecture": "resnet50",
  "training_framework": "PyTorch"
}
```

### 3. Enable Caching

Add to `app.py`:

```python
# Enable caching for faster repeated predictions
@gr.cache
def classify_image_cached(image):
    return classify_image(image)
```

## Monitoring and Analytics

Once deployed, you can monitor your Space:

1. **View Analytics**: Space settings → Analytics
2. **Check Usage**: Monitor API calls and user interactions
3. **View Logs**: Debug issues through the Logs tab
4. **Community**: Enable discussions for user feedback

## Updating Your Space

To update your model or code:

```bash
# Make changes locally
# Update app.py or model weights

# Commit and push
git add .
git commit -m "Update model to version 2"
git push
```

The Space will automatically rebuild and redeploy.

## Best Practices

1. **Model Size**: Keep model files under 1GB for faster loading
2. **Dependencies**: Pin specific versions in `requirements.txt`
3. **Examples**: Provide clear example images
4. **Documentation**: Include clear instructions and model details
5. **Error Handling**: Add try-catch blocks for robust operation
6. **Resource Usage**: Monitor and optimize for the selected hardware tier

## Support

- **Hugging Face Forums**: [discuss.huggingface.co](https://discuss.huggingface.co)
- **Documentation**: [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **Discord**: Join the Hugging Face Discord community

---

Congratulations! Your CIFAR-100 ResNet-50 classifier is now deployed on Hugging Face Spaces! 🎉