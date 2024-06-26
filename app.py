from flask import Flask, render_template, request, jsonify, send_from_directory
import pickle
import cv2
import os
import face_recognition
import numpy as np
import base64
import tempfile
import urllib.parse

app = Flask(__name__)

# Load the encoding file
print("Loading encode file ...")
with open('EncodeFile.p', 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)

encodeListknown, statueIds = encodeListKnownWithIds
print("EncodeFile Loaded.")

# Define the folder containing the videos
videos_folder = 'videos'

# Automatically identify image IDs and map them to corresponding video paths
video_paths = {
    'hatshepsut': 'videos/Hatshepsut.mp4',
    'toot': 'videos/toot.mp4',
    'ramses': 'videos/ramses 2.mp4',
    'nevo': 'videos/nevo.mp4',
    'akhnatoon': 'videos/akhnatoon.mp4',
    'khafraa': 'videos/khafraa.mp4',
    'thutmose': 'videos/Thutmose 3.mp4',
    'horemheb': 'videos/horemheb.mp4',
    'amenhotep': 'videos/amenhotep.mp4',
    'senusret': 'videos/senusret.mp4'
}

face_distance_threshold = 0.4  # Set a threshold for face distance

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(videos_folder, filename)

@app.route('/scan_face', methods=['POST'])
def scan_face():
    data = request.json
    image_data = data['image']
    image_data = image_data.split(",")[1]
    image_data = base64.b64decode(image_data)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
        temp_image.write(image_data)
        temp_image_path = temp_image.name

    img = cv2.imread(temp_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    best_match_index = None
    min_face_distance = float("inf")
    best_face_location = None

    for faceLoc, encodeFace in zip(faceCurFrame, encodeCurFrame):
        faceDis = face_recognition.face_distance(encodeListknown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < min_face_distance:
            min_face_distance = faceDis[matchIndex]
            best_match_index = matchIndex
            best_face_location = faceLoc

    if best_match_index is not None and min_face_distance < face_distance_threshold:
        person_id = statueIds[best_match_index]
        video_path = video_paths.get(person_id, None)
        if video_path:
            encoded_video_path = urllib.parse.quote(video_path)
            print(f"Recognized face as {person_id}, serving video: {encoded_video_path}")
            return jsonify({"video_path": encoded_video_path})
        else:
            print(f"Recognized face as {person_id}, but no video found.")
    
    print("Face not recognized or no video found.")
    return jsonify({"error": "Face not recognized"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)

