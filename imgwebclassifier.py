import os
from io import BytesIO
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
from flask import Flask, request, jsonify

# Initialize the Flask app
app = Flask(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg'}

# Pre-trained model setup
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

# Load the weights metadata
weights = ResNet18_Weights.DEFAULT
categories = weights.meta["categories"]

# Debugging category label, can remove
weights = ResNet18_Weights.DEFAULT
categories = weights.meta.get("categories")
print("Categories loaded:", categories is not None, "Length:", len(categories) if categories else "N/A")
print("Sample category:", categories[0])

# Transformations for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = transform(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model(img)
            _, predicted = outputs.max(1)
        class_id = predicted.item()
        # Map the class ID to its corresponding label
        class_label = categories[class_id]
        return class_id, class_label
    except Exception as e:
        return str(e)

# Check if the file has the correct extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Predict the class of an image
def predict_image(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = transform(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model(img)
            _, predicted = outputs.max(1)
        return predicted.item()
    except Exception as e:
        return str(e)

# Upload endpoint for file submission
@app.route('/upload', methods=['POST'])
def upload_file():
    # Ensure that a file is included in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    # Check if the user submitted an empty file
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # Validate file type
    if file and allowed_file(file.filename):
        try:
            file_bytes = file.read()
            class_id = predict_image(file_bytes)
            return jsonify({"class_id": class_id}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Allowed file types are .png, .jpeg, .jpg"}), 400
    
# New route serving an HTML form with display logic
@app.route('/upload_form')
def upload_form():
    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Image Upload for Classification</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 2rem; }
            #result { margin-top: 1rem; }
        </style>
    </head>
    <body>
        <h1>Upload an Image for Classification</h1>
        <form id="uploadForm">
            <input type="file" name="file" accept="image/png, image/jpeg, image/jpg" required>
            <input type="submit" value="Upload">
        </form>
        <div id="result"></div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(event) {
                event.preventDefault(); // Prevent default form submission (page refresh)

                // Create a FormData object from the form
                const formData = new FormData();
                const fileField = document.querySelector('input[type="file"]');
                formData.append('file', fileField.files[0]);

                try {
                    // POST the image to the /upload endpoint
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });

                    const resultDiv = document.getElementById('result');
                    // If the response is successful, display the classification results
                    if (response.ok) {
                        const data = await response.json();
                        resultDiv.innerHTML = `<p><strong>Class ID:</strong> ${data.class_id}</p>
                                               <p><strong>Class Label:</strong> ${data.class_label}</p>`;
                    } else {
                        // Display error message if response is not OK
                        const errorData = await response.json();
                        resultDiv.innerHTML = `<p style="color: red;">Error: ${errorData.error}</p>`;
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = `<p style="color: red;">Unexpected Error: ${error}</p>`;
                }
            });
        </script>
    </body>
    </html>
    '''

# A simple homepage route
@app.route('/')
def home():
    return "Welcome to the homepage! Use the /upload endpoint to POST your image file.(e.g. URL/upload_form)"

if __name__ == '__main__':
    app.run(debug=True)