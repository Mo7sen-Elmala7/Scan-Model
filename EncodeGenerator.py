import cv2
import face_recognition 
import pickle
import os 
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage




cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL' : "https://facerecognition-29c44-default-rtdb.firebaseio.com/",
    'storageBucket' : "facerecognition-29c44.appspot.com"
})


def upload_image_to_storage(image_path, folder_path):
    filename = os.path.basename(image_path)
    destination_blob_name = f"{folder_path}/{filename}"

    bucket = storage.bucket()
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(image_path)

    return destination_blob_name

def encode_images(images_folder):
    img_list = []
    statue_ids = []

    for person_folder in os.listdir(images_folder):
        person_folder_path = os.path.join(images_folder, person_folder)
        if os.path.isdir(person_folder_path):
            for image_filename in os.listdir(person_folder_path):
                if image_filename.endswith('.jpg') or image_filename.endswith('.png'):
                    image_path = os.path.join(person_folder_path, image_filename)
                    img = cv2.imread(image_path)
                    img_list.append(img)
                    statue_ids.append(person_folder)

                    # Upload image to Firebase Storage
                    image_blob_name = upload_image_to_storage(image_path, person_folder)

    print(statue_ids)
    
    def find_encodings(images_list):
        encode_list = []
        for img in images_list:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(img)
            if len(face_locations) > 0:
                encode = face_recognition.face_encodings(img, face_locations)[0]
                encode_list.append(encode)
            else:
                print(f"No face detected in {img}")
        return encode_list

    print("Encoding Started ...")
    encode_list_known = find_encodings(img_list)
    encode_list_known_with_ids = [encode_list_known, statue_ids]
    print("Encoding Complete.")

    with open("EncodeFileTrial.p", 'wb') as file:
        pickle.dump(encode_list_known_with_ids, file)
    print("File Saved")

# Provide the folder path containing images of all persons
images_folder = 'pics'
encode_images(images_folder)
