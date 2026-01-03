import face_recognition
import os
import pickle

path = "dataset"
known_encodings = []
known_names = []

for person in os.listdir(path):
    person_path = os.path.join(path, person)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person)

# Save encodings for later use
with open("encodings.pkl", "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print(" Encodings saved successfully!")
