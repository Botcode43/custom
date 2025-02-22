import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  
    max_num_hands=1,  
    min_detection_confidence=0.7,  
    min_tracking_confidence=0.8  
)

# Gesture Database
DB_FILE = "gestures.pkl"
gestures_db = {}
if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        gestures_db = pickle.load(f)

# Session State for Gesture Recognition
if "recognized_gestures" not in st.session_state:
    st.session_state.recognized_gestures = []

# Function to normalize hand landmarks
def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    return np.array([(lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]) for lm in landmarks]).flatten()

# Extract hand landmarks from camera frame
def extract_hand_landmarks(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            return normalize_landmarks(landmarks)  # Return first hand found
    return None

# Add a new gesture
def add_new_gesture(name, landmarks):
    gestures_db[name] = landmarks
    with open(DB_FILE, "wb") as f:
        pickle.dump(gestures_db, f)
    st.success(f"✅ Gesture '{name}' added successfully!")
    st.rerun()  # UI Refresh to recognize new gesture instantly

# Recognize gesture in real-time
def recognize_gesture(landmarks):
    if not gestures_db:
        return "Unknown"

    min_distance = float("inf")
    recognized_gesture = None

    for gesture, stored_landmarks in gestures_db.items():
        distance = np.linalg.norm(landmarks - stored_landmarks)
        if distance < min_distance:
            min_distance = distance
            recognized_gesture = gesture

    # Set a threshold for recognition
    threshold = max(0.05, np.mean([np.linalg.norm(lm - landmarks) for lm in gestures_db.values()]) * 0.2)

    # Only return recognized gesture if it's within the threshold
    if min_distance < threshold:
        return recognized_gesture
    else:
        return "Unknown"

# Clear all stored gestures
def clear_gestures():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    st.warning("🚮 All gestures deleted!")
    st.session_state.recognized_gestures = []  # Reset sequence
    st.rerun()
    

# Streamlit UI Layout
st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")
st.title("🖐 Gesture Recognition System")
st.write("Now faster, more accurate, and *real-time gesture detection*! 🚀")

# UI Layout Columns
col1, col2 = st.columns([3, 1])  # Left (Video) - Right (Gesture Sequence)

with col1:
    # Start Camera Button
    start_camera = st.button("📷 Start Camera")
    stop_camera = st.button("❌ Stop Camera")

    if start_camera:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        frame_skip = 3  # Process every 3rd frame for better performance
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            if frame_count % frame_skip == 0:  
                landmarks = extract_hand_landmarks(frame)
                recognized_text = recognize_gesture(landmarks) if landmarks is not None else "Unknown"

                # Display Recognized Gesture
                color = (0, 255, 0) if recognized_text != "Unknown" else (0, 0, 255)
                cv2.putText(frame, f"Gesture: {recognized_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # If valid gesture detected, add to sequence (Avoid "Unknown")
                if recognized_text != "Unknown":
                    if not st.session_state.recognized_gestures or st.session_state.recognized_gestures[-1] != recognized_text:
                        st.session_state.recognized_gestures.append(recognized_text)

                stframe.image(frame, channels="BGR")

            if stop_camera:
                cap.release()
                cv2.destroyAllWindows()
                break

    # Add Gesture Section
    st.subheader("➕ Add a New Gesture")
    gesture_name = st.text_input("Enter Gesture Name:")
    add_button = st.button("💾 Save Gesture")

    if add_button:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            landmarks = extract_hand_landmarks(frame)
            if gesture_name and landmarks is not None:
                add_new_gesture(gesture_name, landmarks)
            else:
                st.error("⚠ Please enter a valid name and show a hand gesture.")
        cap.release()

    # View Saved Gestures
    st.subheader("📋 Saved Gestures")
    if gestures_db:
        for gesture in gestures_db.keys():
            st.write(f"✅ {gesture}")
    else:
        st.info("No gestures saved yet!")

    # Clear Gesture Database
    if st.button("🗑 Clear All Gestures"):
        clear_gestures()

# Right Column: Show Recognized Gesture Sequence
with col2:
    st.subheader("📝 Recognized Gesture Sequence")
    if st.session_state.recognized_gestures:
        st.write(" → ".join(st.session_state.recognized_gestures))
    else:
        st.info("No gestures detected yet!")

st.markdown("---")
st.write("Developed by *Aditya Tripathi* | Powered by Streamlit 🚀")