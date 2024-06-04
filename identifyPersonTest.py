import os
import requests
import base64
from time import sleep

def encode_image_to_base64(file_path):
    """Encode an image file to a base64 string."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def identify_person(url, image_encoded):
    """Send a POST request to the API to identify a person from an image."""
    payload = {"image": image_encoded}
    response = requests.post(url, json=payload)
    return response.json()

def test_identification(api_url, test_folder):
    """Process each image in the test folder, send to API, and check response."""
    results = {}
    for image_name in os.listdir(test_folder):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):  # Ensuring file is an image
            sleep(1)
            image_path = os.path.join(test_folder, image_name)
            image_encoded = encode_image_to_base64(image_path)
            response = identify_person(api_url, image_encoded)
            # Assuming the image name is the person's name without the file extension
            expected_name = image_name.rsplit('.', 1)[0]
            if 'person' in response and 'found' in response and response['found']:
                results[image_name] = (response['person'] == expected_name, response)
            else:
                results[image_name] = (False, response)
    return results

# Usage
api_url = 'http://127.0.0.1:5000/api/v1/identify_person'
test_folder = 'testData'
results = test_identification(api_url, test_folder)
for image_name, (is_correct, response) in results.items():
    print(f"{image_name}:\t\t Match \t{'correct' if is_correct else 'incorrect'}, Response: {response}")
