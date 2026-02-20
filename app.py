import os
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# ------------------------------
# Paths (robust, works for Streamlit)
# ------------------------------
BASE_DIR = os.path.dirname(__file__)  # Folder where app.py is
MODEL_PATH = os.path.join(BASE_DIR, "Emotion_detection_model.h5")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

# ------------------------------
# Load model and cascade
# ------------------------------
model = load_model(MODEL_PATH)
emotion_labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    st.error("Error: Haar Cascade XML file not found or corrupted!")
    st.stop()

# ------------------------------
# Streamlit App UI
# ------------------------------
st.title("🎭 Real-Time Emotion Detection")
run = st.checkbox("Run Webcam")

FRAME_WINDOW = st.image([])

# ------------------------------
# Webcam loop
# ------------------------------
if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access webcam!")
        st.stop()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to read from webcam")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48)) / 255.0
            face = face.reshape(1, 48, 48, 1)

            prediction = model.predict(face, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()