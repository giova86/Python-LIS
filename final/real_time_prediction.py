# real time prediction

import cv2
import time
import mediapipe as mp
import numpy as np
import os
from utils import mediapipe_detection, draw_landmarks, draw_landmarks_custom, draw_limit_rh, draw_limit_lh, check_detection, points_detection
#from keras.models import model_from_json
import pickle
from sklearn import svm


# - INPUT PARAMETERS ------------------------------- #
PATH_MODEL_SVM = '../models/model_svm.sav'
# PATH_MODEL_JSON = '../models/model_rh.json'
# PATH_MODEL_WEIGHTS = '../models/model_rh.h5'
threshold = 0.4
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
labels = np.array(['a', 'b', 'c']) # put the entire alphabet in the future
# -------------------------------------------------- #

# load svm model
model = pickle.load(open(PATH_MODEL_SVM, 'rb'))

# # load json and create model
# json_file = open(PATH_MODEL_JSON, 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# model.load_weights(PATH_MODEL_WEIGHTS)
# print("Loaded model from disk")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=min_detection_confidence,
                          min_tracking_confidence=min_tracking_confidence) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        #frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # make detection
        image, results = mediapipe_detection(frame, holistic)

        color = (0,0,255)
        cv2.rectangle(frame, (0+int(0.03*h),int(h-0.14*h)), (0+int(0.75*h), int(h-0.015*h)), color,-1)

        for i in range(len(labels)):
            cv2.rectangle(frame, (70, 10+ i*int(50)), (70+0, 60+ i*int(50)), color,-1)
            cv2.putText(frame, labels[i], (10, 50+ i*int(50)), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 4, cv2.LINE_AA)

        # perform prediction with relative probability
        if results.right_hand_landmarks:

            #draw_limit_rh(frame, results)

            # uncomment for NN
            # prediction = labels[np.argmax(model.predict(np.array([points_detection(results)])))]

            prediction = model.predict(np.array([points_detection(results)]))[0]
            pred_prob = np.max(model.predict_proba(np.array([points_detection(results)])))

            for i in range(len(labels)):
                cv2.rectangle(frame, (70, 10+ i*int(50)), (70+int(model.predict_proba(np.array([points_detection(results)]))[0][i]*100)*3, 60+ i*int(50)), color,-1)
                cv2.putText(frame, labels[i], (10, 50+ i*int(50)), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 4, cv2.LINE_AA)

            # uncomment for NN
            # for i in range(len(labels)):
            #     cv2.rectangle(frame, (70, 10+ i*int(50)), (70+int(model.predict(np.array([points_detection(results)]))[0][i]*100)*3, 60+ i*int(50)), color,-1)
            #     cv2.putText(frame, labels[i], (10, 50+ i*int(50)), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 4, cv2.LINE_AA)


            # add text with prediction

            if pred_prob > int(threshold):
                cv2.putText(frame, f'{prediction.capitalize()} ({int(pred_prob*100)}%)',
                            (0+int(0.05*h),h-int(0.05*h)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2 ,
                            (0,255,0),
                            4,
                            cv2.LINE_AA)
            else:
                cv2.putText(frame, 'Detecting Hand',
                            (0+int(0.05*h),h-int(0.05*h)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2 ,
                            (0,255,0),
                            4,
                            cv2.LINE_AA)

        else:
                cv2.putText(frame, 'Detecting Hand',
                            (0+int(0.05*h),h-int(0.05*h)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0,255,0),
                            4,
                            cv2.LINE_AA)


        #draw_landmarks_custom(frame, results)

        cv2.imshow('LIS', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
