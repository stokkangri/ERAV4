# Image Question & Answer App

A Flask web application that allows users to upload images and ask questions about them. The app uses Google's Gemini AI to analyze images and provide intelligent answers.

## Files Overview

### `app.py`
The main Flask backend application that handles:

- **File Upload**: Accepts image files (PNG, JPG, JPEG, GIF) through a web form
- **Image Processing**: Reads and encodes uploaded images in base64 format
- **AI Integration**: Sends images directly to Google Gemini AI for analysis
- **Response Generation**: Processes user questions and returns AI-generated answers
- **Static File Serving**: Serves uploaded images back to the frontend
- **Error Handling**: Provides user-friendly error messages for various scenarios

**Key Features:**
- Secure file upload with validation
- Direct image analysis using Gemini AI (no text extraction needed)
- Proper MIME type detection for different image formats
- Environment variable support for API keys

### `index.html`
The frontend template that provides:

- **Upload Form**: File input for image selection with proper validation
- **Question Input**: Large text box for entering questions about the image
- **Results Display**: Shows AI-generated answers and uploaded images
- **Responsive Design**: Clean, user-friendly interface

**Key Features:**
- File type validation (images only)
- Large, accessible text input field
- Dynamic content display based on results
- Proper form encoding for file uploads

## How It Works

1. User uploads an image file through the web interface
2. User enters a question about the image
3. The Flask app saves the image to the `static` folder
4. The image is encoded and sent directly to Google Gemini AI
5. Gemini analyzes the image and answers the question
6. The response and image are displayed to the user

## Requirements

- Python 3.x
- Flask
- Google GenAI library
- Required Python packages: `flask`, `google-genai`, `werkzeug`

## Setup

1. Install dependencies:
   ```bash
   pip install flask google-genai werkzeug
   ```

2. Set up Google Gemini API key (optional - defaults to hardcoded key):
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser to `http://localhost:5000`

## Usage

1. Choose an image file using the file input
2. Type your question in the text box
3. Click "Submit"
4. View the AI-generated answer and your uploaded image

## Supported Image Formats

- PNG
- JPG/JPEG
- GIF

Video: https://youtu.be/P4rnlXEt5cA
