# Animal & File Upload Application

A modern web application built with Flask backend and HTML/CSS/JavaScript frontend that allows users to:
1. Select animals (cat, dog, elephant) and view their images
2. Upload files and get detailed information about them

## Features

- **Animal Selection**: Choose from cat, dog, or elephant with radio buttons
- **File Upload**: Drag & drop or click to browse files
- **Modern UI**: Beautiful gradient design with hover effects and animations
- **Responsive Design**: Works on both desktop and mobile devices
- **Real-time Feedback**: Success/error notifications and loading states

## Project Structure

```
├── app.py                 # Flask backend application
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── style.css         # CSS styling
│   ├── script.js         # JavaScript functionality
│   └── images/           # Animal images (cat.jpg, dog.jpg, elephant.jpg)
├── uploads/              # Directory for uploaded files
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Setup Instructions

### 1. Install Python Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Animal Images
The application now includes actual JPEG images for the animals:
- `cat.jpeg` - Real cat image
- `dog.jpeg` - Real dog image  
- `elephant.jpeg` - Real elephant image

These images are automatically copied to the `static/images/` directory and will be displayed when you select the corresponding animal.

### 3. Run the Application
```bash
python app.py
```

### 4. Access the Application
Open your web browser and navigate to:
```
http://localhost:5000
```

## API Endpoints

- `GET /` - Main page
- `POST /upload` - File upload endpoint
- `GET /animal/<animal_name>` - Get animal image information

## File Upload Support

The application supports various file types including:
- Documents: PDF, DOC, DOCX, TXT
- Images: PNG, JPG, JPEG, GIF
- Spreadsheets: XLS, XLSX
- Archives: ZIP, RAR

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Styling**: Modern CSS with gradients, animations, and responsive design
- **File Handling**: Werkzeug for secure file uploads

## Browser Compatibility

- Chrome (recommended)
- Firefox
- Safari
- Edge

## Short video
https://youtu.be/NlesjsIW-FQ

## License

This project is open source and available under the MIT License.
