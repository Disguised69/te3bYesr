from flask import Flask, Response, request, jsonify, render_template, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sklearn.neighbors import KNeighborsClassifier
import shutil
import csv
import datetime
from time import time
import base64
import threading
import numpy as np
import logging
import cv2
import os
import pandas as pd
from faceEmbedding import resize_image, calculate_face_embedding
from webcamInterface import gen_frames , capture_webcam_image, encode_image_to_base64
from flask_cors import CORS
from faceDetection import extract_face , detect_face, draw_face_box

from config import expected_image_size, embeddings_vector_size, distance_threshold, sleep_between_frames_seconds

last_frame = None
pause_event = False


app = Flask(__name__, static_folder='static')
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///faces.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    image = db.Column(db.LargeBinary, nullable=False)
    embedding = db.Column(db.PickleType, nullable=False)


all_names = []
KNN = None

last_capture = {
    'frame': None,
    'encoded_frame': None
}
last_person = {'name':None,'time':0}
def identify_person(encoded_image):
    with app.app_context():
        persons = Person.query.all()
        if persons:
            try:
                decoded_image = base64.b64decode(encoded_image)
                nparr = np.frombuffer(decoded_image, np.uint8)
                image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                resized_image = resize_image(image_np)
                reference_embedding = calculate_face_embedding(resized_image)
                match_index = KNN.predict(reference_embedding)[0]
                name = all_names[match_index]
                y_probat = KNN.predict_proba(reference_embedding)[0][match_index]
                knn_proba = y_probat

                # Find k-nearest neighbors
                distance, indices = KNN.kneighbors(reference_embedding, 1, True)
                distance = distance[0, 0]

                # Check if the file exists and write headers if it's empty
                file_exists = os.path.isfile('log.csv')
                with open('log.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    if not file_exists or os.stat('log.csv').st_size == 0:
                        # Writing header if file doesn't exist or is empty
                        writer.writerow(['Timestamp ', ' Person Identified ', ' Distance ', ' 4Probability'])

                    # Log the data
                    timestamp = datetime.datetime.now()
                    formatted_timestamp= timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    if((time()-last_person['time']) >10) and (last_person['name'] != name):
                        last_person['time'] = time()
                        last_person['name'] = name if(distance<distance_threshold) else "unknown"
                        writer.writerow([formatted_timestamp, last_person['name'], distance, knn_proba])


                    if distance < distance_threshold:
                        print({'found': True, 'person': name, 'proba': knn_proba, 'distance': distance})
                        return jsonify({'found': True, 'person': name, 'proba': float(knn_proba), 'distance': float(distance)})
                    else:
                        print({'found': False, 'person': name, 'proba': knn_proba, 'distance': distance})
                        return jsonify({'found': False, 'person': name, 'proba': float(knn_proba), 'distance': float(distance)})
            except Exception as e:
                return jsonify({'message': 'No persons found'}), 404

        else:
                return jsonify({'message': 'No persons found'}), 404

            
def continious_capture():
    while True:
        global last_frame

        for frame in gen_frames():

            last_frame = frame
            try:
                image = np.copy(frame)
            except Exception as e:
                print('Frame not available ', e)
                continue

            face_coordinates = detect_face(image)
            if face_coordinates:
                face_image = extract_face(image, face_coordinates)

                # Encode face image to base64
                face_image_encoded = encode_image_to_base64(face_image)

                # Send the encoded image to the API
                response = identify_person(face_image_encoded)

                # Handle the API response
                # Check if response is a Response object
                if isinstance(response, Response):
                    # Extract JSON data from Response object
                    response_data = response.get_json()

                    # Handle the response data
                    if response_data and response_data.get('found'):
                        print(f"Person identified: {response_data['person']} with probability {response_data['proba']}")
                    else:
                        print("Person not identified")

                # Optional: Add some delay to avoid overwhelming the API


@app.route('/pause_capture', methods=['POST'])
def pause_capture():
    global pause_event
    pause_event = True
    print("Face Recognition paused!")
    print(pause_event)
    return jsonify({"status": "Capture paused"}), 200

@app.route('/resume_capture', methods=['POST'])
def resume_capture():
    global pause_event
    pause_event = False
    print("Face Recognition Resumed !")
    return jsonify({"status": "Capture resumed"}), 200

@app.route('/list_persons', methods=['GET'])
def list_persons():
    faces_dir = 'faces'  # Path to the faces directory
    if not os.path.exists(faces_dir):
        return jsonify({"error": "Faces directory not found"}), 404

    persons = [name for name in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, name))]
    return jsonify({"persons": persons})

def retrain_KNN():
    with app.app_context():
        global all_names, KNN
        persons = Person.query.all()
        unique_person_names = Person.query.with_entities(Person.name).distinct()
        names = [name[0] for name in unique_person_names]
        all_names = names
        if persons:
            embeddings = []
            labels = []
            for person in persons:
                person_embedding = np.frombuffer(person.embedding, np.float32).reshape((1, embeddings_vector_size))
                embeddings.append(person_embedding[0])
                labels.append(names.index(person.name))
            X = np.array(embeddings)
            y = np.array(labels)
            print("training data : ", X.shape, y.shape)
            try:
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(X, y)
                KNN = knn
            except Exception as e:
                print("Failed to update KNN ", e)
logging.basicConfig(level=logging.DEBUG)


@app.route('/api/v1/newPerson', methods=['POST'])
def new_person():
    person_name = request.form.get('name')

    if not person_name:
        return jsonify({'error': 'Person name is required'}), 400

    folder_path = os.path.join(os.getcwd(), 'faces', person_name)

    if os.path.exists(folder_path):
        return jsonify({'error': 'Folder already exists'}), 400

    try:
        os.makedirs(folder_path)
        return jsonify({'message': 'Folder created successfully'}), 200
    except Exception as e:
        print('Error creating folder:', e)
        return jsonify({'error': 'Failed to create folder'}), 500


UPLOAD_FOLDER = 'faces'
def saveImage(name,user_folder):
    global last_frame
    try:
        image = np.copy(last_frame)
    except Exception as e:
        print( 'Frame not available ',e)
    face_coordinates = detect_face(image)
    if face_coordinates:
        face_image = extract_face(image, face_coordinates)

        # Load the current count
        count_file = os.path.join(user_folder, 'count.txt')
        if os.path.exists(count_file):
            with open(count_file, 'r') as f:
                count = int(f.read().strip())
        else:
            count = 0

        # Increment the count
        count += 1

        # Save the new count
        with open(count_file, 'w') as f:
            f.write(str(count))

        # Save the face image with the new count
        file_path = os.path.join(user_folder, f'face{count}.png')
        cv2.imwrite(file_path, face_image)
        resized_image = resize_image(face_image)
        # Assuming 'calculate_face_embedding' is a function to calculate the face embedding
        embedding = calculate_face_embedding(resized_image)
        # Assuming 'Person' is the model representing the database table
        new_person = Person(
            name=name,
            image=resized_image,
            embedding=embedding  # Convert embedding to list if needed
        )

        # Assuming 'db' is the database session
        db.session.add(new_person)
        db.session.commit()
        # start train knn thread
        # Create a thread object with your function
        my_thread = threading.Thread(target=retrain_KNN)
        # Start the thread
        my_thread.start()
        return count
    else:
        print('No face detected.')
        return None


@app.route('/capture', methods=['POST'])
def capture_image():
    data = request.json
    name = data['name']

    user_folder = os.path.join(UPLOAD_FOLDER, name)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

        # Save the image and get the updated count
    count = saveImage(name,user_folder)

    if count is not None:
        return jsonify({'status': 'success', 'message': f'Face captured and saved. Total pictures taken: {count}',
                        'count': count}), 200
    else:
        return jsonify({'status': 'error', 'message': 'No face detected or frame not available.'}), 500

@app.route('/log_history')
def log_history():
    try:
        log_df = pd.read_csv('log.csv')
        log_html = log_df.to_html(classes='table table-striped', index=False)
        return render_template('log.html', log_table=log_html)
    except Exception as e:
        return str(e)



# calculate euclidean distance
def euclidean_distance(p, q):
    return np.sqrt(np.sum((p - q) ** 2))


@app.route('/api/v1/person/<name>', methods=['GET'])
def get_person_by_name(name):
    # Querying the database for all entries with the specified name
    persons = Person.query.filter(Person.name == name).all()
    if persons:
        response_data = []
        for person in persons:
            # Read the image file
            nparr = np.frombuffer(person.image, np.uint8).reshape((expected_image_size[0], expected_image_size[1], 3))
            if nparr is None:
                return jsonify({'error': 'Failed to decode the image data'}), 500
            _, buffer = cv2.imencode('.png', nparr)
            if buffer is None:
                return jsonify({'error': 'Failed to encode the image as PNG'}), 500

            png_image = buffer.tobytes()
            encoded_image = base64.b64encode(png_image).decode('utf-8')
            response_data.append({
                'id': person.id,
                'name': person.name,
                'image': encoded_image  # Base64 encoded image data
            })
        return jsonify(response_data), 200
    else:
        return jsonify({'message': 'No persons found with that name'}), 404


@app.route('/api/v1/unique_names', methods=['GET'])
def unique_names():
    # Query the database for unique names using distinct()
    unique_person_names = Person.query.with_entities(Person.name).distinct()
    names = [name[0] for name in unique_person_names]
    if names:
        return jsonify(names), 200
    else:
        return jsonify({'message': 'No persons found'}), 404


@app.route('/api/v1/delete_person/<name>', methods=['DELETE'])
def delete_person(name):
    # Find persons with the given name
    persons_to_delete = Person.query.filter_by(name=name).all()
    if not persons_to_delete:
        return jsonify({'message': 'No person found with that name'}), 404

    # Delete the persons found from the database
    for person in persons_to_delete:
        db.session.delete(person)
    db.session.commit()
    my_thread = threading.Thread(target=retrain_KNN)
    # Start the thread
    my_thread.start()

    # Path to the faces directory
    faces_dir = os.path.join('faces', name)

    # Check if the directory exists and delete it
    if os.path.exists(faces_dir) and os.path.isdir(faces_dir):
        shutil.rmtree(faces_dir)


    return jsonify({'message': f'All entries for {name} were deleted successfully'}), 200



@app.route('/api/v1/delete_all', methods=['DELETE'])
def delete_all():
    # Find persons with the given name
    persons_to_delete = Person.query.all()
    if not persons_to_delete:
        return jsonify({'message': 'No person found with that name'}), 404

    # Delete the persons found from the database
    for person in persons_to_delete:
        db.session.delete(person)
    db.session.commit()

    return jsonify({'message': f'All entries were deleted successfully'}), 200


@app.route('/api/v1/get_all_embeddings', methods=['GET'])
def get_all_persons():
    persons = Person.query.all()
    if persons:
        response_data = []
        for person in persons:
            nparr = np.frombuffer(person.embedding, np.float32).reshape((1, embeddings_vector_size))
            response_data.append({
                'name': person.name,
                'embedding': nparr.tolist()
            })
        return jsonify(response_data), 200
    else:
        return jsonify({'message': 'No persons found'}), 404

@app.route('/api/v1/get_last_frame')
def getLastFrame():
    return jsonify(last_capture), 200

@app.route('/api/v1/video_feed')
def video_feed():
    def generate():
        global last_frame
        while True:
            ret, buffer = cv2.imencode('.jpg', last_frame)
            frameBuffer = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frameBuffer + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def login():
    # Use render_template to serve the HTML file
    return render_template('login.html')

@app.route('/main')
def main():
    # Use render_template to serve the HTML file
    return render_template('main.html')

@app.route('/add_person')
def second():
    return render_template('addPerson.html')

@app.route('/delete_person')
def third():
    return render_template('deletePerson.html')

@app.route('/get_event_log', methods=['GET'])
def get_event_log():
    # Specify the directory where your CSV files are stored
    return send_from_directory(directory='.', path='log.csv', as_attachment=True)



# Create the database tables inside the application context
with app.app_context():
    my_thread = threading.Thread(target=continious_capture)
    my_thread.start()
    db.create_all()
    retrain_KNN()

app.run(host='0.0.0.0', port=3000)




