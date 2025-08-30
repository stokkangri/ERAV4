from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'xls', 'xlsx', 'zip', 'rar'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print(f"Upload request received: {request.method}")
    print(f"Request files: {request.files}")
    print(f"Request form: {request.form}")
    
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    print(f"File received: {file.filename}, Content-Type: {file.content_type}")
    
    if file.filename == '':
        print("No file selected")
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving file to: {file_path}")
        file.save(file_path)
        
        # Get file information
        file_size = os.path.getsize(file_path)
        file_type = file.content_type or 'Unknown'
        
        result = {
            'filename': filename,
            'filesize': file_size,
            'filetype': file_type
        }
        print(f"Upload successful: {result}")
        return jsonify(result)
    
    print(f"File type not allowed: {file.filename}")
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/animal/<animal_name>')
def get_animal_image(animal_name):
    valid_animals = ['cat', 'dog', 'elephant']
    if animal_name.lower() in valid_animals:
        # Check if .jpg exists, otherwise use .jpeg
        jpg_path = f'static/images/{animal_name.lower()}.jpg'
        jpeg_path = f'static/images/{animal_name.lower()}.jpeg'
        
        if os.path.exists(jpg_path):
            image_path = f'/static/images/{animal_name.lower()}.jpg'
        elif os.path.exists(jpeg_path):
            image_path = f'/static/images/{animal_name.lower()}.jpeg'
        else:
            image_path = f'/static/images/{animal_name.lower()}.jpg'  # Default fallback
        
        return jsonify({
            'animal': animal_name.lower(),
            'image_path': image_path
        })
    return jsonify({'error': 'Invalid animal'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
