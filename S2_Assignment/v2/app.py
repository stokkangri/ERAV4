from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import numpy as np

app = Flask(__name__)

# Set up upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Helper function to check allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to upload images
@app.route("/", methods=["GET", "POST"])
def upload_images():
    if request.method == "POST":
        # Get files from the form
        files = request.files.getlist('images')

        # Validate file count (up to 5 images)
        if len(files) == 0 or len(files) > 5:
            return "Please upload between 1 and 5 images."

        # Process images and save them
        uploaded_images = []
        for file in files:
            if file and allowed_file(file.filename):
                # Save to static/images folder
                filename = os.path.join(app.config['STATIC_FOLDER'], file.filename)
                file.save(filename)
                uploaded_images.append(filename)

        # If more than 1 image is uploaded, calculate the mean image
        mean_image_path = None
        if len(uploaded_images) > 1:
            mean_image = calculate_mean_image(uploaded_images)
            # Save the mean image in the static/images folder
            mean_image_path = os.path.join(app.config['STATIC_FOLDER'], 'mean_image.jpg')
            mean_image.save(mean_image_path)

        return render_template('index.html', uploaded_images=uploaded_images, mean_image_path=mean_image_path)

    return render_template('index.html')


# Function to calculate mean image
def calculate_mean_image(image_paths):
    images = [Image.open(img_path) for img_path in image_paths]

    # Find the smallest image size (width, height)
    min_width = min(img.width for img in images)
    min_height = min(img.height for img in images)

    # Resize all images to the smallest size
    resized_images = [img.resize((min_width, min_height)) for img in images]

    # Convert images to numpy arrays
    img_arrays = [np.array(img) for img in resized_images]

    # Calculate the mean along the pixel axis
    mean_array = np.mean(img_arrays, axis=0).astype(np.uint8)

    # Convert the result back to an image
    mean_img = Image.fromarray(mean_array)
    return mean_img


if __name__ == "__main__":
    # Make sure the static/images directory exists
    if not os.path.exists(app.config['STATIC_FOLDER']):
        os.makedirs(app.config['STATIC_FOLDER'])
    
    app.run(debug=True)
