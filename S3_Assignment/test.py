import os
from app import detect_text_from_image, detect_labels_from_image  # Import function to test
from google import genai

# Google GenAI Client setup
client = genai.Client(api_key="AIzaSyA-AwhodgNMy_EkZIxS0Hkj3B1APfNdmQ8")

# Test the corrected function
def test_fixed_function(image_path):
    
    # Test the fixed function
    labels = detect_labels_from_image(image_path)
    print(f"Final result: {labels}")
    
    # Test with Gemini
    if isinstance(labels, list):
        question = "What do you see in this image?"
        ask_gemini(question, labels)

# Test with a simple, known image
def test_vision_api(test_image_path):
    
    print(f"Testing with image: {test_image_path}")
    
    # Check if file exists
    if not os.path.exists(test_image_path):
        print("Test image not found")
        return
    
    # Try to detect labels
    labels = detect_labels_from_image(test_image_path)
    print(f"Result: {labels}")

# Function to query Gemini LLM
def ask_gemini(question, image_labels):
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': "AIzaSyA-AwhodgNMy_EkZIxS0Hkj3B1APfNdmQ8"  # Set the API key in headers
    }

    # Format the labels into a single string
    labels_text = ", ".join(image_labels)
    print (f"Labels {labels_text}")

    # Correct method: using 'models' instead of 'model'
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Here are the labels from the image: {labels_text}\n\nQuestion: {question}"
        )

        if response:
            print("Response from Gemini:", response.text)
        else:
            print("Error: No response from Gemini.")
    except Exception as e:
        print(f"Error calling Gemini: {e}")

# Test detect_labels_from_image function
def test_image_label_extraction(image_path):
    print("Testing image label extraction...")
    
    # Test with a sample image path (replace with an actual image path in your system)
    image_labels = detect_labels_from_image(image_path)
    
    if image_labels:
        print(f"Extracted labels: {image_labels}")
    else:
        print("No labels found in the image.")
    
    return image_labels

import sys
# Main function to test the flow
if __name__ == "__main__":

    #test_vision_api(sys.argv[1])
    #test_fixed_function(sys.argv[1])

    # Test with a sample image file
    image_labels = detect_labels_from_image(sys.argv[1]) # "path_to_your_sample_image.jpg"  # Replace with an actual image path

    if image_labels:
        # Ask a sample question to Gemini
        question = f"What can you infer from the following labels: {', '.join(image_labels)}?"
        ask_gemini(question, image_labels)
