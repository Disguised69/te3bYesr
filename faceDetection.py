import cv2
from webcamInterface import capture_webcam_image

def detect_face(image):
    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are detected, return None
    if len(faces) == 0:
        return None

    # Return the bounding box coordinates of the first detected face
    x, y, w, h = faces[0]
    return (x, y, x + w, y + h)  # Return (x1, y1, x2, y2)

def extract_face(image, face_coordinates):
    # Extract the bounding box coordinates
    x1, y1, x2, y2 = face_coordinates

    # Ensure the coordinates are within the image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    # Extract the face region from the image
    face_image = image[y1:y2, x1:x2]

    return face_image

def draw_face_box(image, face_coordinates):
    # Extract the bounding box coordinates
    x1, y1, x2, y2 = face_coordinates

    # Draw a rectangle around the face
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image
