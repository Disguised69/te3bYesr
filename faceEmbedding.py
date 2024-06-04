from keras_facenet import FaceNet
import cv2
import numpy as np
from config import expected_image_size


def create_an_embedder():
    # Load a pre-trained FaceNet model
    embedder = FaceNet()
    return embedder


def resize_image(image, target_size=expected_image_size):
    # resize the image
    resized_img = cv2.resize(image, target_size)
    resized_img = resized_img.reshape((1, target_size[0], target_size[1], 3))
    return resized_img


def calculate_face_embedding(image, embedder=create_an_embedder()):
    # Generate embeddings for each face in the dataset
    embeddings = embedder.embeddings(image)
    return embeddings

