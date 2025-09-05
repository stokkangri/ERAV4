# Flask Image Mean Calculation Project

This is a simple **Flask web application** that allows users to upload up to **5 images**, calculates the **mean image** of the uploaded images (if more than one image is uploaded), and displays the images along with the mean result.

## Features
- Upload 1 to 5 images via the web interface.
- If more than one image is uploaded, calculate the **mean image** (pixel-wise average).
- Display all uploaded images and the resulting mean image.
- Images are processed and displayed without requiring any manual setup for virtual environments or dependencies (if running the app in the provided setup).

## Requirements
- Python 3.x
- Flask
- Pillow (PIL)
- NumPy

## Installation

### Step 1: Clone the Repository
Clone the project repository to your local machine:
```bash
git clone <your-repository-url>
cd <project-folder>

pip install Flask Pillow numpy
mkdir -p static/images
python app.py
This will start the web server, typically on http://127.0.0.1:5000/.

project-folder/
│
├── app.py                # Flask application
├── templates/
│   └── index.html        # HTML form and template to display images
├── static/
│   └── images/           # Folder for storing uploaded images and mean image
│       └── mean_image.jpg # Resultant mean image (generated)
└── uploads/              # Temporary folder for storing images (optional)


Error Handling
	•	The app requires 1 to 5 images for the operation. If you upload fewer or more than the allowed number, it will display an error.
	•	All images are resized to the smallest size (in terms of width and height) before calculating the mean.

Customization
	•	The application can be customized to work with different image formats (e.g., .png, .jpeg).
	•	You can modify the image mean calculation method (e.g., weighted mean or other mathematical operations).
	•	Modify the HTML form and styling using CSS to suit your needs.

Known Issues
	•	The images must all be in a compatible format (e.g., .png, .jpg, .jpeg). Unsupported formats may cause errors.
	•	If the image sizes differ significantly, resizing will occur before the mean calculation, which may result in some image distortion.

Contributing

Feel free to contribute by creating issues, forking the repository, and submitting pull requests.

License

This project is licensed under the MIT License – see the LICENSE file for details.

---

### Key Sections in the `README.md`:

1. **Project Overview**: Brief introduction to what the project does.
2. **Requirements**: List of dependencies and Python version required.
3. **Installation**: Steps to set up the project, install dependencies, and run the app.
4. **How It Works**: Explanation of how the app processes the images and calculates the mean.
5. **File Structure**: Overview of the project structure.
6. **Customization**: Suggestions for future modifications or extensions.
7. **Contributing**: How others can contribute to the project.
8. **License**: License information (you can adjust this depending on your choice of license).

---

