"""
drowsiness_detector.py
Run with:
    python drowsiness_detector.py          # webcam
    python drowsiness_detector.py --video file.mp4   # video file
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import sys
import time

# ---------- SETTINGS ----------
EAR_THRESHOLD = 0.25        # Eyes closed if below this
EAR_CONSEC_FRAMES = 20      # Frames before triggering alert
MAR_THRESHOLD = 0.6         # Yawn threshold
# ------------------------------

# Landmarks indices from MediaPipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH = [61, 291, 13, 14]

mp_face_mesh = mp.solutions.face_mesh

def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def eye_aspect_ratio(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    p1, p2, p3, p4, p5, p6 = pts
    A = euclidean(p2, p6)
    B = euclidean(p3, p5)
    C = euclidean(p1, p4)
    return (A + B) / (2.0 * C) if C != 0 else 0, pts

def mouth_aspect_ratio(landmarks, indices, w, h):
    left = (int(landmarks[indices[0]].x * w), int(landmarks[indices[0]].y * h))
    right = (int(landmarks[indices[1]].x * w), int(landmarks[indices[1]].y * h))
    top = (int(landmarks[indices[2]].x * w), int(landmarks[indices[2]].y * h))
    bottom = (int(landmarks[indices[3]].x * w), int(landmarks[indices[3]].y * h))
    vertical = euclidean(top, bottom)
    horizontal = euclidean(left, right)
    return vertical / horizontal if horizontal != 0 else 0, (left, right, top, bottom)

def sound_alert():
    """Simple beep alert"""
    try:
        import winsound
        winsound.Beep(2000, 700)
    except:
        sys.stdout.write('\a')
        sys.stdout.flush()

def main(args):
    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        print("ERROR: Cannot open video source")
        return

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    counter = 0
    alarm_on = False
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # EAR
            left_ear, left_pts = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear, right_pts = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            # MAR
            mar, mouth_pts = mouth_aspect_ratio(landmarks, MOUTH, w, h)

            if ear < EAR_THRESHOLD:
                counter += 1
            else:
                counter = 0
                alarm_on = False

            if counter >= EAR_CONSEC_FRAMES:
                alarm_on = True
                sound_alert()

            # Draw eyes
            for p in left_pts + right_pts:
                cv2.circle(frame, p, 1, (0, 255, 0), -1)

            # Draw mouth
            for p in mouth_pts:
                cv2.circle(frame, p, 2, (255, 0, 0), -1)

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if alarm_on:
                cv2.putText(frame, "DROWSINESS ALERT!", (int(w*0.1), int(h*0.2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 3)

            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "YAWN DETECTED", (int(w*0.1), int(h*0.3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,165,255), 2)
        else:
            cv2.putText(frame, "No face detected", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        # FPS
        cur_time = time.time()
        fps = 1 / (cur_time - prev_time) if cur_time != prev_time else 0
        prev_time = cur_time
        cv2.putText(frame, f"FPS: {int(fps)}", (w-120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        cv2.imshow("Drowsiness Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Path to video file", default=None)
    args = parser.parse_args()
    main(args)
