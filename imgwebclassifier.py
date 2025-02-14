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

# Test, can remove
# label = categories[717]
# print(label)  # This will print the human-readable label for class 717.

# Get ImageNet class categories from the weights metadata
categories = weights.meta["categories"]

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
    
@app.route('/upload_form')
# test with URL -> http://127.0.0.1:5000/upload_form
def upload_form():
    return '''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Image Upload</title>
      </head>
      <body>
        <h1>Upload an Image for Classification</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
          <input type="file" name="file" accept="image/png, image/jpeg, image/jpg">
          <input type="submit" value="Upload">
        </form>
      </body>
    </html>
    '''

# A simple homepage route
@app.route('/')
def home():
    return "Welcome to the homepage! Use the /upload endpoint to POST your image file.(e.g. URL/upload_form)"

if __name__ == '__main__':
    app.run(debug=True)