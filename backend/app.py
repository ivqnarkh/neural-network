from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from predict import predict_digit
import matplotlib.pyplot as plt
from PIL import Image
import os
from predict import preprocess_image
import numpy as np

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('model/digit_classifier.keras')

os.makedirs('test_images', exist_ok=True)

@app.route('/')
def home():
    return "Digit Classifier API is running!"

@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'image' not in request.files:
        return jsonify({'error': 'no image provided'}), 400
    
    try:
        #get file from frontend
        image_file = request.files['image']
        
        #process the image
        processed_array = preprocess_image(image_file, invert=True)
        
        #predict digit
        prediction = predict_digit(image_file, model)
        
        return jsonify({'prediction': int(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)