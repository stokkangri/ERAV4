# Cloud Model Hosting Guide for Hugging Face Spaces

Your ResNet-50 model (~95MB) can be hosted in several ways. Here's a comparison and setup guide for each option:

## Option Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Hugging Face Direct** | Simple, integrated, fast loading | Uses Space storage quota | Models < 1GB |
| **Google Drive** | Free 15GB, easy sharing | Requires gdown, may hit quota limits | Personal projects |
| **Dropbox** | Reliable, direct links | 2GB free limit | Production apps |
| **GitHub Releases** | Version control, permanent | 2GB file limit | Open source projects |
| **Hugging Face Hub** | CDN, versioning, LFS support | Separate from Space | Large models, production |

## Method 1: Direct Upload to Hugging Face Space (Recommended)

This is the **recommended approach** for your 95MB model.

### Setup:
```bash
# Install Git LFS
git lfs install

# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/cifar100-resnet50
cd cifar100-resnet50

# Track large files with LFS
git lfs track "*.pth"
git add .gitattributes

# Add your model
cp /path/to/best_model_binary.pth .
git add best_model_binary.pth

# Use the standard app.py
cp /path/to/app.py .

# Commit and push
git commit -m "Add ResNet-50 model"
git push
```

### Advantages:
- ✅ Fastest loading (same server)
- ✅ No external dependencies
- ✅ Automatic CDN distribution
- ✅ Works offline after first load

## Method 2: Google Drive Hosting

Good for larger models or when you want to update models without redeploying.

### Setup:

1. **Upload to Google Drive:**
   - Upload `best_model_binary.pth` to Google Drive
   - Right-click → "Get link" → "Anyone with link can view"
   - Copy the file ID from the URL: `https://drive.google.com/file/d/FILE_ID/view`

2. **Update app_cloud.py:**
```python
MODEL_SOURCE = {
    'type': 'gdrive',
    'id': 'YOUR_FILE_ID_HERE'  # e.g., '1a2B3c4D5e6F7g8H9'
}
```

3. **Add dependency:**
```bash
echo "gdown" >> requirements.txt
```

4. **Deploy to Hugging Face:**
```bash
# Use app_cloud.py instead of app.py
cp app_cloud.py app.py
git add app.py requirements.txt
git commit -m "Use Google Drive model hosting"
git push
```

### Advantages:
- ✅ Can update model without redeploying
- ✅ 15GB free storage
- ✅ Model cached after first download

### Limitations:
- ⚠️ May hit download quota (100GB/day)
- ⚠️ Slower initial load
- ⚠️ Requires internet connection

## Method 3: Dropbox Hosting

More reliable than Google Drive for production use.

### Setup:

1. **Upload to Dropbox:**
   - Upload `best_model_binary.pth` to Dropbox
   - Create share link
   - Change `?dl=0` to `?dl=1` in the URL

2. **Update app_cloud.py:**
```python
MODEL_SOURCE = {
    'type': 'url',
    'url': 'https://www.dropbox.com/s/xxxxx/best_model_binary.pth?dl=1'
}
```

3. **Deploy:**
```bash
cp app_cloud.py app.py
git add app.py
git commit -m "Use Dropbox model hosting"
git push
```

### Advantages:
- ✅ Reliable, no quota issues
- ✅ Direct download links
- ✅ Good for production

## Method 4: GitHub Releases

Best for versioned, open-source models.

### Setup:

1. **Create GitHub Release:**
```bash
# In your GitHub repo
git tag v1.0
git push --tags

# Go to GitHub → Releases → Create Release
# Upload best_model_binary.pth as release asset
```

2. **Get direct download URL:**
```
https://github.com/USERNAME/REPO/releases/download/v1.0/best_model_binary.pth
```

3. **Update app_cloud.py:**
```python
MODEL_SOURCE = {
    'type': 'url',
    'url': 'https://github.com/USERNAME/REPO/releases/download/v1.0/best_model_binary.pth'
}
```

### Advantages:
- ✅ Version control
- ✅ Permanent URLs
- ✅ Fast CDN

## Method 5: Hugging Face Model Hub

Best for production and model versioning.

### Setup:

1. **Create Model Repository:**
```bash
# Install huggingface-hub
pip install huggingface-hub

# Login
huggingface-cli login

# Create model repo
from huggingface_hub import create_repo, upload_file

create_repo("YOUR_USERNAME/cifar100-resnet50", repo_type="model")

# Upload model
upload_file(
    path_or_fileobj="best_model_binary.pth",
    path_in_repo="pytorch_model.bin",
    repo_id="YOUR_USERNAME/cifar100-resnet50"
)
```

2. **Update app to use Hub:**
```python
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/cifar100-resnet50",
    filename="pytorch_model.bin"
)
```

### Advantages:
- ✅ Professional solution
- ✅ Model versioning
- ✅ Download statistics
- ✅ Model cards

## Optimization Tips

### 1. Model Compression (Reduce Size)
```python
# Save with optimization
torch.save(
    model.state_dict(),
    'model_optimized.pth',
    _use_new_zipfile_serialization=True
)

# Or use quantization
import torch.quantization as quantization
quantized_model = quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

### 2. Lazy Loading
```python
# Load model only when needed
@st.cache_resource
def load_model():
    return load_model_from_cloud()
```

### 3. Progressive Loading UI
```python
with gr.Row():
    status = gr.Textbox("Loading model...", label="Status")
    
# Update status during loading
status.value = "Downloading model (95MB)..."
```

## Deployment Checklist

For your 95MB ResNet-50 model, I recommend:

### ✅ Quick Start (Direct Upload):
```bash
# 1. Clone space
git clone https://huggingface.co/spaces/YOUR_USERNAME/cifar100-resnet50

# 2. Add files
cp app.py requirements.txt best_model_binary.pth cifar100-resnet50/

# 3. Setup Git LFS
cd cifar100-resnet50
git lfs track "*.pth"

# 4. Push
git add .
git commit -m "Deploy CIFAR-100 classifier"
git push
```

### ✅ Production Setup (Cloud Hosting):
```bash
# 1. Upload model to cloud (choose one):
#    - Google Drive (free, 15GB)
#    - Dropbox (reliable, 2GB free)
#    - GitHub Releases (versioned)

# 2. Use app_cloud.py with your chosen source
cp app_cloud.py app.py

# 3. Configure MODEL_SOURCE in app.py

# 4. Deploy
git add .
git commit -m "Deploy with cloud model"
git push
```

## Testing Before Deployment

```python
# Test locally with cloud model
python app_cloud.py

# Verify model loads correctly
# Check loading time
# Test predictions
```

## Monitoring

After deployment, monitor:
- Loading times
- Memory usage
- Download failures
- Cache hit rates

## Summary

For your 95MB model:
- **Development**: Use direct upload (Method 1)
- **Production**: Use Hugging Face Hub (Method 5)
- **Personal Projects**: Use Google Drive (Method 2)
- **Quick Sharing**: Use GitHub Releases (Method 4)

The `app_cloud.py` file supports all methods - just change the `MODEL_SOURCE` configuration!