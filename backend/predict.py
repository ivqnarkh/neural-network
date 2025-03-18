import numpy as np
from PIL import Image
import io

def preprocess_image(file_stream, invert=True):
    """Process image with optional inversion for web inputs"""
    img = Image.open(file_stream).convert('L').resize((28, 28))
    img_array = np.array(img)
    
    if invert:  # Only invert web inputs, not test data
        img_array = 255.0 - img_array  # Invert colors
    
    img_array = img_array / 255.0  # Normalize to [0,1]
    return img_array.reshape(1, 28, 28, 1)

def predict_digit(image_file, model):
    processed_image = preprocess_image(image_file)
    predictions = model.predict(processed_image)
    return np.argmax(predictions[0])