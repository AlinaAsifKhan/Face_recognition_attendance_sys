from ultralytics import YOLO
import cv2
import face_recognition
import pickle
import numpy as np
import csv
from datetime import datetime
import os


# Load YOLO face detection model
model = YOLO("model.pt")

# Load known face encodings
with open("encodings.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

cap = cv2.VideoCapture(0)

marked_names = set()
today = datetime.now().strftime("%Y-%m-%d")

if os.path.exists("attendance.csv"):
    with open("attendance.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            name, date = row[0], row[1]
            if date == today:
                marked_names.add(name)

while True:
    ret, frame = cap.read()
    if not ret:

        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            enc = face_recognition.face_encodings(face_rgb)

            name = "Unknown"
            if len(enc) > 0:
                distances = face_recognition.face_distance(known_encodings, enc[0])
                best_idx = np.argmin(distances)
                if distances[best_idx] < 0.5:  
                    name = known_names[best_idx]
                    if name not in marked_names:
                        now = datetime.now()
                        date_today = now.strftime("%Y-%m-%d")
                        time_now = now.strftime("%H:%M:%S")

                        with open("attendance.csv", "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([name, date_today, time_now])

                        marked_names.add(name)


            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
