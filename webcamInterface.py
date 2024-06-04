import cv2

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
    #face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    global last_frame
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        else:
            last_frame["x"] = frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            faces = []
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


