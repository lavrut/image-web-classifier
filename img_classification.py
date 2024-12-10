import os
from io import BytesIO
from PIL import Image
import torch
from torchvision import models, transforms
from flask import Flask, request, jsonify

# Initialize the Flask app
app = Flask(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg'}

# Pre-trained model setup
model = models.resnet18(pretrained=True)
model.eval()

# Transformations for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file and allowed_file(file.filename):
        try:
            file_bytes = file.read()
            class_id = predict_image(file_bytes)
            return jsonify({"class_id": class_id}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Allowed file types are .png, .jpeg, .jpg"}), 400

if __name__ == '__main__':
    app.run(debug=True)
