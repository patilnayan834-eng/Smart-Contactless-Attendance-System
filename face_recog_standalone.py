import cv2
import numpy as np
import face_recognition
import os
import sqlite3
from datetime import datetime

dataset_path = 'dataset'
conn = sqlite3.connect('database.db', check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    name TEXT,
                    time TEXT
                )''')
conn.commit()

def mark_attendance(name):
    time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO attendance (name, time) VALUES (?, ?)", (name, time_now))
    conn.commit()
    print(f"Attendance marked for {name} at {time_now}")

def load_images():
    names, encodings = [], []
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_encoding = face_recognition.face_encodings(img_rgb)
                if img_encoding:
                    encodings.append(img_encoding[0])
                    names.append(folder)
    return encodings, names

def recognize_faces():
    known_encodings, known_names = load_images()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cam.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_small)
        encodes = face_recognition.face_encodings(rgb_small, faces)

        for encode_face, face_loc in zip(encodes, faces):
            matches = face_recognition.compare_faces(known_encodings, encode_face)
            face_distance = face_recognition.face_distance(known_encodings, encode_face)
            match_index = np.argmin(face_distance)

            if matches[match_index]:
                name = known_names[match_index].upper()
                mark_attendance(name)
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow("Smart Attendance System - Press Q to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
