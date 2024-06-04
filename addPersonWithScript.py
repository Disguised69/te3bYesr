import os
import requests
import base64

def encode_image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def add_person(url, image_encoded, name):
    payload = {
        "image": image_encoded,
        "name": name
    }
    response = requests.post(url, json=payload)
    return response.json()

def process_folder(base_path, url):
    # Loop through all directories in the base path
    for person_name in os.listdir(base_path):
        person_path = os.path.join(base_path, person_name)
        if os.path.isdir(person_path):
            # Loop through each image in the person's folder
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                if os.path.isfile(image_path):
                    print("-> add", image_name, image_path)
                    image_encoded = encode_image_to_base64(image_path)
                    result = add_person(url, image_encoded, person_name)
                    print(f"Added {person_name}, Response: {result}")

# Usage
api_url = 'http://127.0.0.1:5000/api/v1/add_person'
faces_folder = 'faces'
process_folder(faces_folder, api_url)
