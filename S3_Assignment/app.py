from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import base64
from google import genai
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set up the upload folder and allowed extensions
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Google GenAI Client setup
# Note: In production, use environment variables for API keys
api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyA-AwhodgNMy_EkZIxS0Hkj3B1APfNdmQ8')
client = genai.Client(api_key=api_key)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to serve uploaded files
@app.route('/static/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Note: We're now using Gemini directly for image analysis instead of Google Vision API

# Route to upload an image and ask a question
@app.route("/", methods=["GET", "POST"])
def upload_and_ask():
    if request.method == "POST":
        try:
            # Get the uploaded image file
            file = request.files['image']
            question = request.form['question']
            
            if not file or file.filename == '':
                return render_template("index.html", answer="No file selected", image_url=None)
            
            if not allowed_file(file.filename):
                return render_template("index.html", answer="Invalid file type. Please upload a PNG, JPG, JPEG, or GIF file.", image_url=None)
            
            # Save the uploaded image
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Step 1: Read the image file and encode it
            with open(file_path, "rb") as image_file:
                image_data = image_file.read()
            
            # Get the correct MIME type
            file_extension = filename.split('.')[-1].lower()
            mime_type_map = {
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg', 
                'png': 'image/png',
                'gif': 'image/gif'
            }
            mime_type = mime_type_map.get(file_extension, 'image/jpeg')

            # Step 2: Create the content to send to Gemini LLM with the actual image
            contents = [
                {
                    "text": f"Question: {question}\n\nPlease analyze this image and answer the question based on what you can see in it."
                },
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(image_data).decode('utf-8')
                    }
                }
            ]

            # Step 3: Generate the response from Gemini LLM
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp", 
                contents=contents
            )
            
            # Step 4: Extract and display the response
            answer = response.text
            image_url = url_for('static', filename=filename)
            return render_template("index.html", answer=answer, image_url=image_url)
            
        except Exception as e:
            print(f"Error processing request: {e}")
            return render_template("index.html", answer=f"Error processing your request: {str(e)}", image_url=None)

    return render_template("index.html", answer=None, image_url=None)

if __name__ == "__main__":
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    app.run(debug=True)
