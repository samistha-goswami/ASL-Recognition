import pickle
import cv2
import numpy as np
import streamlit as st
from collections import deque
import base64

# Load the model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'model.p' exists.")
    st.stop()

# Labels dictionary
labels_dict = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f',
    6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l',
    12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r',
    18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x',
    24: 'y', 25: 'z', 26: '0', 27: '1', 28: '2',
    29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: 'I love You', 37: 'yes', 38: 'No', 39: 'Hello', 40: 'Thanks',
    41: 'Sorry', 43: 'space'
}

# Function to add background image from a local file
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background image
add_bg_from_local("C:\\Users\\RAMNARESH\\Downloads\\Extra\\Extra\\BGI1.jpg")

# Streamlit UI
st.markdown("""<style>.upper-middle-content { display: flex; ... }</style>""", unsafe_allow_html=True)

st.markdown('<div class="upper-middle-content">', unsafe_allow_html=True)
st.title("ASL Recognition")
start_button = st.button("Start Video")
stop_button = st.button("Stop Video")
clear_button = st.button("Clear Text")
st.markdown('</div>', unsafe_allow_html=True)

text_output = st.empty()
video_placeholder = st.empty()

# Shared variables
prediction_queue = deque(maxlen=5)
running = False

# Hand detection logic
def detect_hands(frame, hand_model):
    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (256, 256), (0, 0, 0), swapRB=True, crop=False)
    hand_model.setInput(blob)
    detections = hand_model.forward()

    hands_landmarks = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Confidence threshold
            x_min = int(detections[0, 0, i, 3] * frame.shape[1])
            y_min = int(detections[0, 0, i, 4] * frame.shape[0])
            x_max = int(detections[0, 0, i, 5] * frame.shape[1])
            y_max = int(detections[0, 0, i, 6] * frame.shape[0])
            
            # Extract hand region
            hand_roi = frame[y_min:y_max, x_min:x_max]
            hands_landmarks.append((hand_roi, (x_min, y_min, x_max, y_max)))

    return hands_landmarks

# Function to run the ASL recognition
def run_asl_recognition():
    global running
    cap = cv2.VideoCapture(0)

    # Load pre-trained hand detection model
    hand_model = cv2.dnn.readNetFromCaffe("hand_deploy.prototxt", "hand_model.caffemodel")

    if not cap.isOpened():
        st.error("Cannot access the camera. Please check your device.")
        running = False
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)
        hands_landmarks = detect_hands(frame, hand_model)

        for hand_roi, bbox in hands_landmarks:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Flatten ROI data for model input
            data_aux = hand_roi.flatten()

            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
            except Exception as e:
                st.error(f"Prediction error: {e}")
                predicted_character = None

            if predicted_character:
                prediction_queue.append(predicted_character)

            if len(prediction_queue) == prediction_queue.maxlen:
                stabilized_character = max(set(prediction_queue), key=prediction_queue.count)
                text_output.text(f"Prediction: {stabilized_character}")

        # Display video frame
        video_placeholder.image(frame, channels="BGR")

    cap.release()

# Start/Stop button logic
if start_button:
    running = True
    run_asl_recognition()

if stop_button:
    running = False

if clear_button:
    text_output.text("Text Cleared")
