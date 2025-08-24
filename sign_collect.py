import cv2
import numpy as np
import os
import mediapipe as mp
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image_bgr, model):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, results

def draw_landmarks(image, results):
    # Face
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    # Pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # Left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # Right Hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results) -> np.ndarray:
    pose = np.zeros(33*4, dtype=np.float32)     # 132
    face = np.zeros(468*3, dtype=np.float32)    # 1404
    lh = np.zeros(21*3, dtype=np.float32)       # 63
    rh = np.zeros(21*3, dtype=np.float32)       # 63

    if results.pose_landmarks:
        pose = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark], dtype=np.float32).flatten()
    
    if results.face_landmarks:
        face = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.face_landmarks.landmark], dtype=np.float32).flatten()

    if results.left_hand_landmarks:
        lh = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()

    if results.right_hand_landmarks:
        rh = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()

    return np.concatenate([pose, face, lh, rh]) # 1662 ค่าต่อเฟรม

ACTION = "hello"
SEQ_IDX = 0
SEQ_LEN = 30
OUT_DIR = os.path.join("MP_Data", ACTION, str(SEQ_IDX))
os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
recording = False
frame_i = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)

        if recording:
            kpts = extract_keypoints(results)
            np.save(os.path.join(OUT_DIR, f"{frame_i}.npy"), kpts)
            frame_i += 1
            cv2.putText(image, f"Recording {ACTION}/{SEQ_IDX} {frame_i}/{SEQ_LEN}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            if frame_i >= SEQ_LEN:
                recording = False
                frame_i = 0
                print(f"Sequence saved to: {OUT_DIR}")
        else: 
            cv2.putText(image, "Press 'r' to start recording", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Record 1 sequence', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r') and not recording:
            recording = True
            frame_i = 0

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()