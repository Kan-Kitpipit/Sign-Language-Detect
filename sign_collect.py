import cv2
import numpy as np
import os
import mediapipe as mp
import time
from pathlib import Path

#--------------- CONFIG ---------------
DATA_PATH = Path('MP_Data')
SEQUENCES_LENGTH = 30
#--------------------------------------

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ฟังก์ชันตรวจจับ Mediapipe
def mediapipe_detection(image_bgr, model):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, results

# ฟังก์ชันวาด landmarks
def draw_landmarks(image, results):
    # Face
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    # Pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # Left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # Right Hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# ฟังก์ชันดึง keypoints
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

#---------------------- MAIN ----------------------
def main():
    actions = []
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        current_action = None
        recording = False
        sequence_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            status_text = f"Action: {current_action if current_action else 'None'} | Sequence: {sequence_count} | Recording: {recording}"
            cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if recording else (0, 0, 255), 2)

            cv2.imshow('Sign Language Data Collector', image)

            # Key event
            key = cv2.waitKey(10) & 0xFF

            # กด 'n' เพื่อเพิ่ม action ใหม่
            if key == ord('n'):
                new_action = input("Enter new action name: ").strip()
                if new_action:
                    actions.append(new_action)
                    current_action = new_action
                    Path(DATA_PATH / current_action).mkdir(parents=True, exist_ok=True)
                    print(f"[INFO] Action: {current_action}")

            # กด 's' เพื่อเริ่มการบันทึก
            if key == ord('s') and current_action:
                if not recording:
                    print(f"[INFO] Start record in 3 seconds...")
                    time.sleep(3)
                    print(f"[INFO] Start Recording...")
                    recording = True
                    frame_count = 0
                    existing = list((DATA_PATH / current_action).glob('*'))
                    sequence_id = len(existing)
                    seq_path = DATA_PATH / current_action / f"Sequence_{sequence_id}"
                    seq_path.mkdir(parents=True, exist_ok=True)

                else:
                    recording = False
                    print(f"[INFO] Stop Recording...")

            # กด 'q' เพื่อออกจากโปรแกรม
            if key == ord('q'):
                break

            if recording and current_action:
                keypoints = extract_keypoints(results)
                npy_path = seq_path / f"frame_{frame_count}.npy"
                np.save(npy_path, keypoints)
                frame_count += 1

                if frame_count >= SEQUENCES_LENGTH:
                    print(f"[INFO] Recorded {SEQUENCES_LENGTH} frames for action '{current_action}'.")
                    recording = False

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()