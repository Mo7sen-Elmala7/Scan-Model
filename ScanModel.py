import pickle
import cv2
import os
import face_recognition
import numpy as np
import cvzone
from moviepy.editor import VideoFileClip

# Capture video from the camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

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
    'toot'      : 'videos/toot.mp4',
    'ramses'    : 'videos/ramses 2.mp4',
    'nevo'      : 'videos/nevo.mp4',
    'akhnatoon' : 'videos/akhnatoon.mp4',
    'khafraa'   : 'videos/khafraa.mp4',
    'thutmose'  : 'videos/Thutmose 3.mp4',
    'horemheb'  : 'videos/horemheb.mp4',
    'amenhotep' : 'videos/amenhotep.mp4',
    'senusret'  : 'videos/senusret.mp4'
}
for video_filename in os.listdir(videos_folder):
    if video_filename.endswith('.mp4'):
        person_id = os.path.splitext(video_filename)[0]
        video_path = os.path.join(videos_folder, video_filename)
        video_paths[person_id] = video_path

play_video = False
face_distance_threshold = 0.4  # Set a threshold for face distance


while True:
    success, img = cap.read()
    if not success:
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    best_match_index = None
    min_face_distance = float("inf")
    best_face_location = None

    for faceLoc, encodeFace in zip(faceCurFrame, encodeCurFrame):
        faceDis = face_recognition.face_distance(encodeListknown, encodeFace)
        matchIndex = np.argmin(faceDis)
        
        print("faceDis", faceDis)
        print("matchIndex", matchIndex)
        print("matched statue ID", statueIds[matchIndex])

        if faceDis[matchIndex] < min_face_distance:
            min_face_distance = faceDis[matchIndex]
            best_match_index = matchIndex
            best_face_location = faceLoc

    if best_match_index is not None and min_face_distance < face_distance_threshold:
        y1, x2, y2, x1 = best_face_location
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        bbox = x1, y1, x2 - x1, y2 - y1
        img = cvzone.cornerRect(img, bbox, rt=0)

        cv2.imshow("Cam", img)
        cv2.waitKey(20)

        play_video = True

        person_id = statueIds[best_match_index]
        print(f"Detected person ID: {person_id}")

        # Get the video path for the detected person
        video_path = video_paths.get(person_id, None)
        print(f"Video path: {video_path}")

        if video_path:
            video_clip = VideoFileClip(video_path)
            video_clip = video_clip.resize(width=640, height=480)
            video_clip.preview()
            video_clip.close()

    cv2.imshow("Cam", img)
    cv2.waitKey(2)

    if play_video:
        person_id = statueIds[best_match_index]
        # Get the video path for the detected person
        video_path = video_paths.get(person_id, None)
        if video_path:
            video_clip = VideoFileClip(video_path)
            video_clip = video_clip.resize(width=640, height=480)
            video_clip.preview()
            video_clip.close()
        play_video = False

