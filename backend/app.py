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
        image_file = request.files['image']
        
        # Save raw image
        #raw_path = os.path.join('test_images', 'raw_image.png')
        #image_file.save(raw_path)
        #print(f"Saved raw image to: {raw_path}")
        
        # Reset file pointer after saving raw image
        #image_file.seek(0)  # <-- THIS IS CRUCIAL
        
        # Process image
        processed_array = preprocess_image(image_file, invert=True)
        
        # Convert processed array to image
        #denormalized = (processed_array[0,:,:,0] * 255).astype(np.uint8)
        #img = Image.fromarray(denormalized)
        
        # Save processed image
        #processed_path = os.path.join('test_images', 'processed_image.png')
        #img.save(processed_path)
        #print(f"Saved processed image to: {processed_path}")
        
        # Make prediction
        prediction = predict_digit(image_file, model)  # Pass array, not file
        
        return jsonify({'prediction': int(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)