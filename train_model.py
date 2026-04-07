import cv2
import os
import numpy as np
from PIL import Image

dataset_path = "dataset"
trainer_path = "trainer/trainer.yml"
os.makedirs("trainer", exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

faces = []
ids = []
label_map = {}
current_id = 0

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    label_map[current_id] = person_name

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = Image.open(image_path).convert("L")
        img_np = np.array(img, "uint8")

        faces_detected = detector.detectMultiScale(img_np)
        for (x, y, w, h) in faces_detected:
            faces.append(img_np[y:y+h, x:x+w])
            ids.append(current_id)

    current_id += 1

recognizer.train(faces, np.array(ids))
recognizer.save(trainer_path)

with open("trainer/labels.txt", "w") as f:
    for id_, name in label_map.items():
        f.write(f"{id_},{name}\n")

print("Model trained successfully")