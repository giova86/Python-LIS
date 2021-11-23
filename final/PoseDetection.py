# libraries
import cv2
import time
import mediapipe as mp
import numpy as np
import os
from utils import mediapipe_detection, draw_landmarks, draw_landmarks_custom

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        black = np.zeros((1080, 1920, 3))
        ret, frame = cap.read()
        #frame = cv2.flip(frame,1)

        # make detection
        image, results = mediapipe_detection(frame, holistic)
        points_rh_x = []
        points_rh_y = []
        if results.right_hand_landmarks:
            for i in range(21):
                points_rh_x.append(results.right_hand_landmarks.landmark[i].x)
                points_rh_y.append(results.right_hand_landmarks.landmark[i].y)

        draw_landmarks_custom(frame, results)

        cv2.imshow('Face, Hands and Pose detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
