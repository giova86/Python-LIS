# methods
import cv2
import time
import mediapipe as mp
import numpy as np
import os
import numpy as np
import pandas as pd

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_landmarks_custom(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color=(0,0,255),thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(0,0,255),thickness=1, circle_radius=1),
                             )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,110,10),thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121),thickness=1, circle_radius=1),
                             )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(255,0,0),thickness=3, circle_radius=5),
                             mp_drawing.DrawingSpec(color=(255,0,0),thickness=3, circle_radius=5),
                             )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(255,0,0),thickness=3, circle_radius=5),
                             mp_drawing.DrawingSpec(color=(255,0,0),thickness=3, circle_radius=5),
                             )

def points_detection(results):
    xMax = max([i.x for i in results.right_hand_landmarks.landmark])
    xMin = min([i.x for i in results.right_hand_landmarks.landmark])
    yMax = max([i.y for i in results.right_hand_landmarks.landmark])
    yMin = min([i.y for i in results.right_hand_landmarks.landmark])
    rh = np.array([[points.x, points.y, points.z] for points in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    for i in np.arange(0, 63, 3):
        rh[i]=(rh[i]-xMin)/(xMax-xMin)
    for i in np.arange(1, 63, 3):
        rh[i]=(rh[i]-yMin)/(yMax-yMin)

    # lh = np.array([[points.x, points.y, points.z] for points in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # po = np.array([[points.x, points.y, points.z] for points in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(99)
    # return np.concatenate([lh, rh, po])

    return rh
