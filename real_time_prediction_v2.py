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
from argparse import ArgumentParser


# - INPUT PARAMETERS ------------------------------- #
parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="ML_model", default='models/model_svm_all.sav',
                    help="PATH of model FILE.", metavar="FILE")
parser.add_argument("-t", "--threshold", dest="threshold_prediction", default=0.5, type=float,
                    help="Threshold for prediction. A number between 0 and 1. default is 0.5")
parser.add_argument("-dc", "--det_conf", dest="min_detection_confidence", default=0.5, type=float,
                    help="Threshold for prediction. A number between 0 and 1. default is 0.5")
parser.add_argument("-tc", "--trk_conf", dest="min_tracking_confidence", default=0.5, type=float,
                    help="Threshold for prediction. A number between 0 and 1. default is 0.5")
args = parser.parse_args()
# -------------------------------------------------- #


# load svm model
model = pickle.load(open(args.ML_model, 'rb'))
labels = np.array(model.classes_) # put the entire alphabet in the future

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
words = []
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=args.min_detection_confidence,
                          min_tracking_confidence=args.min_tracking_confidence) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()

        #frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # make detection
        image, results = mediapipe_detection(frame, holistic)

        color = (0,0,0)
        #cv2.rectangle(frame, (0+int(0.03*h),int(h-0.14*h)), (0+int(0.75*h), int(h-0.015*h)), color,-1)
        cv2.rectangle(frame, (0, 0),
                             (int(w*0.18), int(h-h*0.12)), (255,255,255),-1)


        for i in range(len(labels)):
#            cv2.rectangle(frame, (90, 10+ i*int(50)), (90, 60+ i*int(50)), color,-1)
            cv2.putText(frame, labels[i], (50, (i+1)*int(h/(len(labels)+4))), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (90, (i)*int(h/(len(labels)+4))+30),
                                 (90, (i+1)*int(h/(len(labels)+4)) ), color,-1)

        # perform prediction with relative probability
        if results.right_hand_landmarks:

            # draw_limit_rh(frame, results)

            # uncomment for NN
            # prediction = labels[np.argmax(model.predict(np.array([points_detection(results)])))]

            prediction = model.predict(np.array([points_detection(results)]))[0]
            pred_prob = np.max(model.predict_proba(np.array([points_detection(results)])))

            for i in range(len(labels)):
#                cv2.rectangle(frame, (70, 10+ i*int(50)), (70+int(model.predict_proba(np.array([points_detection(results)]))[0][i]*100)*3, 60+ i*int(50)), color,-1)
                cv2.putText(frame, labels[i], (50, (i+1)*int(h/(len(labels)+4))), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (90, (i)*int(h/(len(labels)+4))+30),
                                     (90+int(model.predict_proba(np.array([points_detection(results)]))[0][i]*100)*2, (i+1)*int(h/(len(labels)+4)) ), color,-1)

            # uncomment for NN
            # for i in range(len(labels)):
            #     cv2.rectangle(frame, (70, 10+ i*int(50)), (70+int(model.predict(np.array([points_detection(results)]))[0][i]*100)*3, 60+ i*int(50)), color,-1)
            #     cv2.putText(frame, labels[i], (10, 50+ i*int(50)), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 4, cv2.LINE_AA)


            # add text with prediction
            if pred_prob > args.threshold_prediction:
                cv2.putText(frame, f'{prediction.capitalize()} ({int(pred_prob*100)}%)',
                            (0+int(0.05*h),h-int(0.05*h)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2 ,
                            (0,255,0),
                            2,
                            cv2.LINE_AA)
            elif pred_prob < 0.3:
                cv2.putText(frame, 'I am not sure...',
                            (0+int(0.05*h),h-int(0.05*h)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2 ,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA)
            else:
                cv2.putText(frame, f'Maybe {prediction.capitalize()} ({int(pred_prob*100)}%)',
                            (0+int(0.05*h),h-int(0.05*h)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2 ,
                            (45, 255, 255),
                            2,
                            cv2.LINE_AA)

        else:
                cv2.putText(frame, 'Detecting Hand...',
                            (w-int(0.5*h),int(0.05*h)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0,0,0),
                            2,
                            cv2.LINE_AA)


        #draw_landmarks_custom(frame, results)

        cv2.imshow('LIS: real time alphabet prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
