from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the pre-trained model
MODEL_PATH = r'E:\image_classification_backend\models\my_image_classifier1.keras'
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Define class names (CIFAR-10)
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
TARGET_IMAGE_SIZE = (32, 32)  # CIFAR-10 image size

@app.route('/')
def index():
    return '''
    <h1>Image Classification API</h1>
    <form method="POST" action="/classify" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Classify Image">
    </form>
    '''

@app.route('/classify', methods=['POST'])
def classify_image_api():
    if model is None:
        return jsonify({'error': 'Model not loaded or failed to load'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Load and preprocess the image
        image = Image.open(file.stream).convert('RGB')
        image = image.resize(TARGET_IMAGE_SIZE)
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(image_array)
        predictions = predictions[0]  # Extract single prediction

        results = [
            {'label': CLASS_NAMES[i], 'confidence': float(predictions[i])}
            for i in range(len(CLASS_NAMES))
        ]
        # Sort by confidence in descending order
        results = sorted(results, key=lambda x: x['confidence'], reverse=True)

        return jsonify({'predictions': results})
    except Exception as e:
        logging.error(f"Image preprocessing failed: {e}")
        return jsonify({'error': f'Image preprocessing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
