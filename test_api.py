import requests
import os

def test_api(image_folder):
    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            path = os.path.join(image_folder, filename)
            with open(path, 'rb') as f:
                response = requests.post(
                    'http://localhost:5000/predict',
                    files={'image': f}
                )
            print(f"{filename}: {response.json()}")


test_api(r'model_training\test_images')