# Image Classification Flask App

This project is a simple image classification web application built with Flask and PyTorch. It leverages a pre-trained ResNet18 model from `torchvision` to classify uploaded images (PNG, JPEG, or JPG). The app returns both the numeric class ID and the corresponding human-readable label from the ImageNet dataset.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The application provides an `/upload` endpoint where users can POST image files. The image is processed, classified using the pre-trained ResNet18 model, and the prediction (including both the class index and the label) is returned as a JSON response.

## Features

- **Image Upload:** Accepts PNG, JPEG, and JPG files.
- **Image Preprocessing:** Resizes, normalizes, and transforms images for model input.
- **Pre-trained Model:** Uses ResNet18 with pretrained weights.
- **Classification Output:** Returns the predicted ImageNet class index and corresponding label.
- **Simple Flask Server:** Easy to run locally for testing and development.

## Prerequisites

- **Python 3.7+**
- **Pip** (Python package installer)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name

2. **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate

3.	**Install the required packages:**

    ```bash
    pip install -r requirements.txt

4. **How to run**
    ```bash
    yourapp.python

#### Notes: 
- The server will start on http://127.0.0.1:5000.
- To upload an image to the model http://127.0.0.1:5000/upload_form 
- Use the /upload endpoint to POST an image. For example, using curl:

    ```bash
    curl -X POST -F "file=@/path/to/your/image.jpg" http://127.0.0.1:5000/upload

- Expected JSON Response:

```JSON
    {
  "class_id": "some numeric index will populate here",
  "class_label": "your_label_here"
}
```

### Access the Homepage

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) to see a basic welcome message with instructions.

---

### Code Structure

- **app.py**: Main Flask application file containing:
  - Import statements for necessary libraries.
  - Pre-trained ResNet18 model and image transformation setup.
  - Route definitions:
    - `/` : A simple homepage.
    - `/upload` : Handles image uploads and classification.
- **requirements.txt**: Contains a list of required packages (Flask, torch, torchvision, pillow).

---

### Customization

- **HTML Upload Form:**  
  If you want a browser-based interface to upload images, you can add an HTML form route (e.g., `/upload_form`) to your Flask app. This route can render an HTML page with a file input and submit button.

- **Production Deployment:**  
  For production, consider using a WSGI server (like Gunicorn or uWSGI) and disable Flask’s debug mode. Update configurations accordingly.

---

### Troubleshooting

- **404 Error on Homepage:**  
  The homepage (`/`) only displays a simple welcome message. To perform image classification, make sure to POST your image to the `/upload` endpoint.

- **File Type Issues:**  
  Ensure that you are uploading an image with one of the allowed extensions (`png`, `jpeg`, `jpg`). Other file types will result in an error.

- **Model Prediction Errors:**  
  If the model fails to predict correctly, verify that the image is properly formatted and that all dependencies are installed correctly.

---

### License

This project is licensed under the **MIT License**.

---

### Acknowledgments

- **Flask** for the web framework.
- **PyTorch** and **torchvision** for providing pre-trained models.
- **ImageNet** for the classification labels.