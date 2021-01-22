import cv2
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from webcam import webcam


def main():
    model = load_model("mask_detection.h5")
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    labels_dict = {0: " MASK", 1: " NO MASK"}
    color_dict = {0: (0, 255, 0), 1: (255, 0, 0)}

    st.title("Webcam Live Feed")
    run = st.checkbox('Run')
    frame_window = st.image([])

    while run:
        captured_image = webcam()
        frame = captured_image.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            face_img = gray[y:y + w, x:x + w]
            resized = cv2.resize(face_img, (100, 100))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 100, 100, 1))
            result = model.predict(reshaped)

            label = np.argmax(result, axis=1)[0]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color_dict[label], 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), color_dict[label], -1)
            cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        frame_window.image(frame)


if __name__ == "__main__":
    main()
