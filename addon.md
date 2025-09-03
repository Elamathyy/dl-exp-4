Code: import cv2 import numpy as np from keras.models import load_model from sklearn.preprocessing import Normalizer from sklearn.metrics.pairwise import cosine_similarity import pickle import os import gdown

-------------------------
Step 1: Download FaceNet model if not present
-------------------------
facenet_path = "facenet_keras.h5" if not os.path.exists(facenet_path): print("Downloading FaceNet model...") url = "https://drive.google.com/uc?id=1sRWXsKNbWwNv1-4k1ajwY5TzTW4uJ4sQ" gdown.download(url, facenet_path, quiet=False)

-------------------------
Step 2: Load FaceNet model
-------------------------
print("Loading FaceNet model...") model = load_model(facenet_path) l2_normalizer = Normalizer('l2')

-------------------------
Step 3: Preprocess face function
-------------------------
def preprocess_face(img): img = cv2.resize(img, (160, 160)) img = img.astype('float32') mean, std = img.mean(), img.std() img = (img - mean) / std return np.expand_dims(img, axis=0)

-------------------------
Step 4: Load stored embeddings
-------------------------
if os.path.exists("face_embeddings.pkl"): with open("face_embeddings.pkl", "rb") as f: known_embeddings, known_names = pickle.load(f) print("Loaded stored embeddings.") else: known_embeddings, known_names = [], [] print("No stored embeddings found! Please register faces first.")

-------------------------
Step 5: Webcam loop
-------------------------
cap = cv2.VideoCapture(0)

while True: ret, frame = cap.read() if not ret: break

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    face = frame[y:y+h, x:x+w]
    face_pp = preprocess_face(face)
    embedding = model.predict(face_pp)[0]
    embedding = l2_normalizer.transform([embedding])[0]

    # Compare with DB
    name = "Unknown"
    max_sim = 0
    for db_emb, db_name in zip(known_embeddings, known_names):
        sim = cosine_similarity([embedding], [db_emb])[0][0]
        if sim > 0.7 and sim > max_sim:  # threshold = 0.7
            name = db_name
            max_sim = sim

    # Draw results
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, name, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

cv2.imshow("Face Recognition - Press Q to Exit", frame)

if cv2.waitKey(1) & 0xFF == ord('q'):
    break




screenshot:<img width="529" height="445" alt="addon ex4" src="https://github.com/user-attachments/assets/bd16ce31-b601-40b1-82a3-1af1cbe6c935" />
