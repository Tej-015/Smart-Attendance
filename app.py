from flask import Flask, request, render_template_string
import cv2
import os
import sqlite3
import numpy as np
from PIL import Image
from datetime import datetime

app = Flask(__name__)

DB_NAME = "attendance.db"
DATASET_PATH = "dataset"
TRAINER_DIR = "trainer"
TRAINER_PATH = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_PATH = os.path.join(TRAINER_DIR, "labels.txt")

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)

# -----------------------------
# DATABASE SETUP
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_db()

# -----------------------------
# HTML TEMPLATES
# -----------------------------
HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Attendance Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #0f172a, #1e3a8a, #7c3aed);
            background-size: 300% 300%;
            animation: gradientMove 12s ease infinite;
            color: white;
            min-height: 100vh;
        }

        @keyframes gradientMove {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .container {
            width: 92%;
            max-width: 1200px;
            margin: auto;
            padding: 24px 0;
        }

        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding: 16px;
            background: rgba(255,255,255,0.08);
            border-radius: 16px;
            backdrop-filter: blur(10px);
        }

        .logo h2 { font-size: 26px; }
        .logo p { font-size: 13px; opacity: 0.8; }

        .hero {
            text-align: center;
            margin-bottom: 24px;
        }

        .hero h1 {
            font-size: 36px;
            margin-bottom: 10px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }

        .stat-card {
            padding: 20px;
            border-radius: 16px;
            background: rgba(255,255,255,0.1);
            text-align: center;
        }

        .stat-card h3 {
            margin-bottom: 8px;
            font-size: 24px;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .glass {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 16px;
            backdrop-filter: blur(8px);
        }

        .glass h3 {
            margin-bottom: 14px;
        }

        input {
            width: 100%;
            padding: 12px;
            border-radius: 10px;
            border: none;
            margin-bottom: 10px;
            outline: none;
            font-size: 15px;
        }

        .btn {
            display: block;
            width: 100%;
            padding: 12px;
            border-radius: 10px;
            margin-top: 10px;
            text-align: center;
            color: white;
            text-decoration: none;
            font-weight: bold;
            border: none;
            cursor: pointer;
            font-size: 15px;
        }

        .btn-primary { background: #2563eb; }
        .btn-success { background: #22c55e; }
        .btn-warning { background: #f59e0b; }
        .btn-dark { background: #374151; }

        .btn:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }

        .feature-section {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-top: 20px;
        }

        .feature-card {
            padding: 16px;
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            text-align: center;
        }

        .feature-card h4 {
            margin-bottom: 6px;
        }

        .quick-info p {
            margin-bottom: 10px;
            line-height: 1.6;
        }

        .footer {
            text-align: center;
            margin-top: 20px;
            opacity: 0.8;
        }

        @media(max-width:900px){
            .stats-grid, .feature-section, .main-grid {
                grid-template-columns: 1fr;
            }

            .hero h1 {
                font-size: 28px;
            }
        }
    </style>
</head>
<body>
    <div class="container">

        <div class="navbar">
            <div class="logo">
                <h2>Smart Attendance System</h2>
                <p>Face Recognition Dashboard</p>
            </div>
        </div>

        <div class="hero">
            <h1>Smart Attendance Dashboard</h1>
        </div>

        <div class="stats-grid">
            <div class="stat-card"><h3>AI</h3><p>Recognition</p></div>
            <div class="stat-card"><h3>Fast</h3><p>Processing</p></div>
            <div class="stat-card"><h3>Secure</h3><p>No Proxy</p></div>
            <div class="stat-card"><h3>Auto</h3><p>Attendance</p></div>
        </div>

        <div class="main-grid">
            <div class="glass">
                <h3>Register Student</h3>
                <form action="/capture" method="post">
                    <input type="text" name="name" placeholder="Enter student name" required>
                    <button type="submit" class="btn btn-primary">Capture Face</button>
                </form>

                <a href="/train" class="btn btn-success">Train Model</a>
                <a href="/start" class="btn btn-warning">Start Attendance</a>
                <a href="/view" class="btn btn-dark">View Records</a>
            </div>

            <div class="glass quick-info">
                <h3>Quick Info</h3>
                <p><b>Step 1:</b> Register student face images</p>
                <p><b>Step 2:</b> Train the face recognition model</p>
                <p><b>Step 3:</b> Start live attendance</p>
                <p><b>Step 4:</b> View attendance records</p>
            </div>
        </div>

        <div class="feature-section">
            <div class="feature-card"><h4>Detection</h4><p>Detect faces live</p></div>
            <div class="feature-card"><h4>Recognition</h4><p>Identify students</p></div>
            <div class="feature-card"><h4>Auto Marking</h4><p>Save date and time</p></div>
            <div class="feature-card"><h4>No Duplicates</h4><p>One entry per day</p></div>
        </div>

        <div class="footer">
            Smart Attendance System • Hackathon Project
        </div>

    </div>
</body>
</html>
"""

MESSAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Status Message</title>
    <style>
        body {
            margin: 0;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #4facfe, #00f2fe);
        }

        .card {
            width: 90%;
            max-width: 520px;
            background: rgba(255,255,255,0.18);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 35px;
            border-radius: 20px;
            text-align: center;
            color: white;
            box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        }

        .card h2 {
            margin-bottom: 15px;
            font-size: 28px;
        }

        .card p {
            font-size: 18px;
            margin-bottom: 25px;
        }

        .btn {
            display: inline-block;
            background: white;
            color: #0072ff;
            text-decoration: none;
            padding: 12px 24px;
            border-radius: 12px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="card">
        <h2>System Message</h2>
        <p>{{ message }}</p>
        <a href="/" class="btn">Go Back Home</a>
    </div>
</body>
</html>
"""

VIEW_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Attendance Records</title>
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #1d2671, #c33764);
            min-height: 100vh;
            color: white;
            padding: 30px 0;
        }

        .container {
            width: 92%;
            max-width: 1100px;
            margin: auto;
        }

        .card {
            background: rgba(255,255,255,0.14);
            padding: 25px;
            border-radius: 20px;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 20px;
            gap: 12px;
        }

        .header h2 {
            margin: 0;
            font-size: 30px;
        }

        .btn {
            display: inline-block;
            text-decoration: none;
            background: #ffffff;
            color: #333;
            padding: 10px 18px;
            border-radius: 12px;
            font-weight: bold;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }

        .stat-box {
            background: rgba(255,255,255,0.16);
            padding: 18px;
            border-radius: 16px;
            text-align: center;
        }

        .stat-box h3 {
            margin: 0;
            font-size: 26px;
        }

        .stat-box p {
            margin: 8px 0 0;
            opacity: 0.9;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            overflow: hidden;
            border-radius: 14px;
            background: white;
            color: #333;
        }

        th, td {
            padding: 14px;
            text-align: center;
        }

        th {
            background: #222;
            color: white;
        }

        tr:nth-child(even) {
            background: #f3f3f3;
        }

        tr:hover {
            background: #e9f2ff;
        }

        .empty {
            text-align: center;
            background: white;
            color: #444;
            padding: 20px;
            border-radius: 12px;
        }

        @media(max-width: 768px) {
            .stats {
                grid-template-columns: 1fr;
            }

            .header h2 {
                font-size: 24px;
            }

            table {
                font-size: 14px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="header">
                <h2>Attendance Records</h2>
                <a href="/" class="btn">Back to Home</a>
            </div>

            <div class="stats">
                <div class="stat-box">
                    <h3>{{ records|length }}</h3>
                    <p>Total Records</p>
                </div>
                <div class="stat-box">
                    <h3>Live</h3>
                    <p>Attendance Status</p>
                </div>
                <div class="stat-box">
                    <h3>Smart</h3>
                    <p>Face Recognition Enabled</p>
                </div>
            </div>

            {% if records %}
            <table>
                <tr>
                    <th>Student Name</th>
                    <th>Date</th>
                    <th>Time</th>
                </tr>
                {% for row in records %}
                <tr>
                    <td>{{ row[0] }}</td>
                    <td>{{ row[1] }}</td>
                    <td>{{ row[2] }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <div class="empty">No attendance records found.</div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM attendance WHERE student_name = ? AND date = ?",
        (name, today)
    )
    existing = cursor.fetchone()

    if not existing:
        cursor.execute(
            "INSERT INTO attendance (student_name, date, time) VALUES (?, ?, ?)",
            (name, today, now_time)
        )
        conn.commit()
        status = "Attendance Marked"
    else:
        status = "Already Marked"

    conn.close()
    return status


def train_model():
    if not hasattr(cv2, "face"):
        return False, "OpenCV face module not found. Please run with correct OpenCV contrib setup."

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = []
    ids = []
    label_map = {}
    current_id = 0

    if not os.path.exists(DATASET_PATH):
        return False, "Dataset folder not found."

    people = os.listdir(DATASET_PATH)
    if not people:
        return False, "No student images found. Please capture faces first."

    for person_name in people:
        person_path = os.path.join(DATASET_PATH, person_name)

        if not os.path.isdir(person_path):
            continue

        image_files = os.listdir(person_path)
        if not image_files:
            continue

        label_map[current_id] = person_name

        for image_name in image_files:
            image_path = os.path.join(person_path, image_name)

            try:
                img = Image.open(image_path).convert("L")
                img_np = np.array(img, "uint8")
            except Exception:
                continue

            detected_faces = detector.detectMultiScale(img_np)

            if len(detected_faces) == 0:
                faces.append(img_np)
                ids.append(current_id)
            else:
                for (x, y, w, h) in detected_faces:
                    faces.append(img_np[y:y + h, x:x + w])
                    ids.append(current_id)

        current_id += 1

    if len(faces) == 0:
        return False, "No valid faces detected for training."

    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINER_PATH)

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for id_, name in label_map.items():
            f.write(f"{id_},{name}\n")

    return True, "Model trained successfully."


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template_string(HOME_HTML)


@app.route("/capture", methods=["POST"])
def capture():
    name = request.form["name"].strip()

    if not name:
        return render_template_string(MESSAGE_HTML, message="Invalid student name.")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO students (name) VALUES (?)", (name,))
    conn.commit()
    conn.close()

    person_path = os.path.join(DATASET_PATH, name)
    os.makedirs(person_path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        return render_template_string(MESSAGE_HTML, message="Camera not accessible.")

    count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if count >= 30:
                break

            count += 1
            face = gray[y:y + h, x:x + w]
            file_path = os.path.join(person_path, f"{count}.jpg")
            cv2.imwrite(file_path, face)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(
            frame,
            f"Images Captured: {count}/30",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Capturing Faces - Press ESC to Stop", frame)

        key = cv2.waitKey(1)
        if key == 27 or count >= 30:
            break

    cam.release()
    cv2.destroyAllWindows()

    return render_template_string(
        MESSAGE_HTML,
        message=f"Face data captured for {name}. Total images: {count}"
    )


@app.route("/train")
def train():
    success, msg = train_model()
    return render_template_string(MESSAGE_HTML, message=msg)


@app.route("/start")
def start():
    if not hasattr(cv2, "face"):
        return render_template_string(
            MESSAGE_HTML,
            message="OpenCV face module not found. Please run with correct OpenCV contrib setup."
        )

    if not os.path.exists(TRAINER_PATH) or not os.path.exists(LABELS_PATH):
        return render_template_string(
            MESSAGE_HTML,
            message="Please train the model first."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)

    labels = {}
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            id_, name = line.strip().split(",", 1)
            labels[int(id_)] = name

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        return render_template_string(MESSAGE_HTML, message="Camera not accessible.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]

            try:
                label_id, confidence = recognizer.predict(face)
            except Exception:
                continue

            if confidence < 60:
                name = labels.get(label_id, "Unknown")
                attendance_status = mark_attendance(name)
                text = f"{name} | {attendance_status} | Conf: {round(confidence, 2)}"
                color = (0, 255, 0)
            else:
                text = f"Unknown | Conf: {round(confidence, 2)}"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("Live Attendance - Press ESC to Stop", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

    return render_template_string(
        MESSAGE_HTML,
        message="Attendance session completed."
    )


@app.route("/view")
def view():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT student_name, date, time FROM attendance ORDER BY id DESC")
    records = cursor.fetchall()
    conn.close()

    return render_template_string(VIEW_HTML, records=records)


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)