import pickle
import cv2
import mediapipe as mp
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

# Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, max_num_hands=1)

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
st.markdown("""
    <style>
        /* Align all content to the upper middle */
        .upper-middle-content {{
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            align-items: center;
            height: 100vh; /* Full viewport height */
            text-align: center;
            margin-top: 50px; /* Push content downward slightly */
        }}
        /* Style title */
        .upper-middle-content h1 {{
            color: black;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px;
        }}
        /* Style buttons */
        .stButton button {{
            background-color: black !important;
            color: white !important;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 20px;
            margin: 10px;
        }}
    </style>
""", unsafe_allow_html=True)

# Create upper-middle content
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

# Function to run the ASL recognition
def run_asl_recognition():
    global running
    cap = cv2.VideoCapture(0)

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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Prepare data for model prediction
                data_aux = []
                x_, y_ = [], []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

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
