from flask import Flask, render_template, Response, request, redirect, url_for
import sqlite3
import cv2
import numpy as np
import os
from datetime import datetime
import threading

app = Flask(__name__)

db_path = 'database.db'
dataset_path = 'dataset'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
current_recognized_name = None
recognition_lock = threading.Lock()
recognizer = None
label_map = {}
last_dataset_count = -1

# Create table if not exists
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    name TEXT,
                    time TEXT
                )''')
conn.commit()
conn.close()


def mark_attendance(name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute("SELECT * FROM attendance WHERE name = ? AND time LIKE ?", (name, today + '%'))
    if cursor.fetchone():
        conn.close()
        return
    time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO attendance (name, time) VALUES (?, ?)", (name, time_now))
    conn.commit()
    conn.close()
    print(f"Attendance marked for {name} at {time_now}")


def prepare_training_data():
    images = []
    labels = []
    label_map = {}
    next_label = 0
    if not os.path.exists(dataset_path):
        return images, labels, label_map

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        label = next_label
        label_map[label] = folder
        next_label += 1
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                continue
            x, y, w, h = faces[0]
            face = gray[y:y+h, x:x+w]
            images.append(face)
            labels.append(label)

    return images, labels, label_map


def count_dataset_images():
    if not os.path.exists(dataset_path):
        return 0
    total = 0
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    total += 1
    return total


def retrain_recognizer():
    global recognizer, label_map, last_dataset_count
    images, labels, new_label_map = prepare_training_data()
    last_dataset_count = count_dataset_images()
    if len(images) == 0:
        recognizer = None
        label_map = {}
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    label_map = new_label_map


retrain_recognizer()


def maybe_retrain_recognizer():
    global last_dataset_count
    current_count = count_dataset_images()
    if current_count != last_dataset_count:
        retrain_recognizer()


def generate_frames():
    global current_recognized_name
    maybe_retrain_recognizer()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera not available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        if recognizer is not None and len(faces) > 0:
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                if face.size == 0:
                    continue
                try:
                    label, confidence = recognizer.predict(face)
                except cv2.error:
                    continue
                name = label_map.get(label, 'Unknown')
                if confidence < 90:
                    display_name = f"{name}"
                    with recognition_lock:
                        current_recognized_name = name
                    mark_attendance(name)
                else:
                    display_name = 'Unknown'
                    with recognition_lock:
                        current_recognized_name = None
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, display_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        elif len(faces) > 0:
            with recognition_lock:
                current_recognized_name = None
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def show_records():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance ORDER BY time DESC")
    rows = cursor.fetchall()
    conn.close()
    records = [{'name': row[0], 'timestamp': row[1]} for row in rows]
    with recognition_lock:
        current_name = current_recognized_name
    return render_template('index.html', records=records, recognized_name=current_name)


@app.route('/retrain_recognizer', methods=['POST'])
def retrain_recognizer_route():
    retrain_recognizer()
    return redirect(url_for('show_records'))


@app.route('/add_attendance', methods=['POST'])
def add_attendance():
    name = request.form['name']
    mark_attendance(name)
    return redirect(url_for('show_records'))


@app.route('/delete_attendance', methods=['POST'])
def delete_attendance():
    name = request.form['name']
    timestamp = request.form['timestamp']
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance WHERE name = ? AND time = ?", (name, timestamp))
    conn.commit()
    conn.close()
    return redirect(url_for('show_records'))


@app.route('/delete_all_attendance', methods=['POST'])
def delete_all_attendance():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance")
    conn.commit()
    conn.close()
    return redirect(url_for('show_records'))


@app.route('/confirm_recognition', methods=['POST'])
def confirm_recognition():
    global current_recognized_name
    name = request.form.get('name')
    if name:
        mark_attendance(name)
    with recognition_lock:
        current_recognized_name = None
    return redirect(url_for('show_records'))


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
