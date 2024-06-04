import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return '<h1>Welcome Home</h1>'

@app.route('/addPerson')
def add_person():
    return render_template('index.html')

@app.route('/newPerson', methods=['POST'])
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

if __name__ == '__main__':
    app.run(debug=True)
