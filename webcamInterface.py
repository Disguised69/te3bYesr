import cv2
import base64

last_frame = {}

def capture_webcam_image(cap, stream=False):
    # Initialize the webcam
    if not cap:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit(1)
        return None  # Return None if the webcam couldn't be accessed

    # Capture one frame
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Failed to capture image.")
        return None

    # Release the webcam
    if not stream:
        cap.release()

    # Return the captured image frame
    return frame

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device")
        return
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        else:
            yield frame
    cap.release()

def encode_image_to_base64(image):
    """Encode an image array to a base64 string."""
    # Convert the image array to JPEG format (or any other desired format)
    _, encoded_image = cv2.imencode('.jpg', image)
    encoded_image_bytes = encoded_image.tobytes()
    encoded_image_str = base64.b64encode(encoded_image_bytes).decode('utf-8')
    return encoded_image_str

