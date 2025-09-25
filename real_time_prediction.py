# real time prediction - improved layout with external sidebar

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


def create_sidebar_panel(labels, predictions=None, width=300, height=600):
    """Create external sidebar panel with predictions"""
    # Create sidebar image
    sidebar = np.zeros((height, width, 3), dtype=np.uint8)
    sidebar.fill(45)  # Dark gray background
    
    # Title section
    title_height = 80
    cv2.rectangle(sidebar, (0, 0), (width, title_height), (30, 30, 30), -1)
    cv2.putText(sidebar, "SIGN RECOGNITION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(sidebar, "Live Prediction", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    
    # Draw separator line
    cv2.line(sidebar, (10, title_height), (width-10, title_height), (100, 100, 100), 2)
    
    # Calculate spacing for labels and bars
    available_height = height - title_height - 40
    item_height = available_height // len(labels)
    bar_height = min(20, item_height - 30)
    
    for i, label in enumerate(labels):
        y_start = title_height + 20 + i * item_height
        
        # Calculate positions for side-by-side layout
        label_x = 15
        label_y = y_start + (item_height // 2) + 5  # Center vertically in the available space
        
        # Progress bar - positioned next to label
        bar_x = 60  # Space for letter + some margin
        bar_y = y_start + (item_height // 2) - (bar_height // 2)  # Center bar vertically
        bar_width = width - bar_x - 30  # Leave space for percentage text
        
        # Label text - positioned to the left of the bar
        cv2.putText(sidebar, label.upper(), (label_x+10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Background bar
        cv2.rectangle(sidebar, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
        cv2.rectangle(sidebar, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 1)
        
        # Fill bar based on prediction
        if predictions is not None:
            progress = predictions[i]
            fill_width = int(bar_width * progress)
            
            # Color based on confidence
            if progress > 0.7:
                color_fill = (0, 255, 0)  # Green
            elif progress > 0.4:
                color_fill = (0, 165, 255)  # Orange
            else:
                color_fill = (0, 100, 200)  # Red
                
            if fill_width > 2:
                cv2.rectangle(sidebar, (bar_x + 1, bar_y + 1), 
                            (bar_x + fill_width - 1, bar_y + bar_height - 1), color_fill, -1)
            
            # Percentage text - positioned to the right of the bar
            percentage_text = f"{int(progress * 100)}%"
            text_x = bar_x + bar_width + 10
            text_y = label_y  # Align with the letter text
            cv2.putText(sidebar, percentage_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    return sidebar


def draw_prediction_overlay(frame, prediction, confidence, threshold, h, w):
    """Draw the main prediction overlay on video"""
    panel_height = int(h * 0.14)
    panel_y = h - panel_height - 10
    panel_x = 10
    panel_width = w - 20
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Border
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (100, 100, 100), 2)
    
    # Status and prediction text
    text_y = panel_y + 25
    
    if prediction:
        if confidence > threshold:
            status_text = "DETECTED"
            status_color = (0, 255, 0)
            pred_color = (255, 255, 255)
            main_text = f"Letter: {prediction.upper()}"
            conf_text = f"Confidence: {int(confidence * 100)}%"
        elif confidence > 0.3:
            status_text = "MAYBE"
            status_color = (0, 165, 255)
            pred_color = (200, 200, 200)
            main_text = f"Maybe: {prediction.upper()}"
            conf_text = f"Confidence: {int(confidence * 100)}%"
        else:
            status_text = "UNCERTAIN"
            status_color = (0, 0, 255)
            pred_color = (150, 150, 150)
            main_text = "Uncertain..."
            conf_text = f"Low confidence: {int(confidence * 100)}%"
    else:
        status_text = "SEARCHING"
        status_color = (255, 255, 0)
        pred_color = (200, 200, 200)
        main_text = "Show your RIGHT hand..."
        conf_text = "Position right hand in camera view"
    
    # Draw texts
    cv2.putText(frame, status_text, (panel_x + 15, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
    cv2.putText(frame, main_text, (panel_x + 15, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, pred_color, 2, cv2.LINE_AA)
    cv2.putText(frame, conf_text, (panel_x + 15, text_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def draw_hand_detection_indicator(frame, has_hand, w, h):
    """Draw hand detection indicator"""
    indicator_size = 15
    indicator_x = w - 40
    indicator_y = 30
    
    if has_hand:
        cv2.circle(frame, (indicator_x, indicator_y), indicator_size, (0, 255, 0), -1)
        cv2.putText(frame, "RIGHT HAND", (indicator_x - 80, indicator_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (indicator_x, indicator_y), indicator_size, (0, 0, 255), -1)
        cv2.putText(frame, "NO HAND", (indicator_x - 65, indicator_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


# Load SVM model
model = pickle.load(open(args.ML_model, 'rb'))
labels = np.array(model.classes_)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Sidebar dimensions
sidebar_width = 350  # Increased width for side-by-side layout
sidebar_height = 600

with mp_holistic.Holistic(min_detection_confidence=args.min_detection_confidence,
                          min_tracking_confidence=args.min_tracking_confidence) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
            
        # DO NOT flip frame to keep right hand as right hand
        h, w, c = frame.shape

        # Make detection
        image, results = mediapipe_detection(frame, holistic)

        prediction = None
        pred_prob = 0
        predictions_array = None

        # Check for RIGHT hand specifically
        has_right_hand = results.right_hand_landmarks is not None
        
        if has_right_hand:
            # Get prediction using right hand
            prediction = model.predict(np.array([points_detection(results)]))[0]
            pred_prob = np.max(model.predict_proba(np.array([points_detection(results)])))
            predictions_array = model.predict_proba(np.array([points_detection(results)]))[0]

        # Create sidebar panel
        sidebar = create_sidebar_panel(labels, predictions_array, sidebar_width, sidebar_height)
        
        # Resize frame to match sidebar height if needed
        if h != sidebar_height:
            aspect_ratio = w / h
            new_width = int(sidebar_height * aspect_ratio)
            frame = cv2.resize(frame, (new_width, sidebar_height))
            h, w, c = frame.shape

        # Draw overlays on main video
        draw_prediction_overlay(frame, prediction, pred_prob, args.threshold_prediction, h, w)
        draw_hand_detection_indicator(frame, has_right_hand, w, h)
        
        # Add title to main video
        cv2.rectangle(frame, (0, 0), (w, 35), (20, 20, 20), -1)
        cv2.putText(frame, "Sign Language Recognition - Camera Feed", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Optional: Draw hand landmarks on main video
        # if has_right_hand:
        #     draw_landmarks_custom(frame, results)

        # Combine main video and sidebar horizontally
        combined_frame = np.hstack([frame, sidebar])
        
        # Show combined window
        cv2.imshow('LIS: Sign Language Recognition System', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()