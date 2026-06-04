# real time prediction
# Layout: "type specimen × strumento di precisione"
# Palette ink + cream + amber, mirino e spettro di confidenza — coerente con la versione React.

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


# ── PALETTE (BGR) ──────────────────────────────────────────── #
INK        = (10, 11, 12)
INK_1      = (13, 15, 17)
PANEL      = (20, 24, 28)
TRACK      = (34, 38, 42)
LINE       = (44, 48, 52)
CREAM      = (215, 234, 243)
CREAM_DIM  = (144, 168, 179)
CREAM_MUTE = (87, 102, 109)
CREAM_FAINT= (58, 66, 72)
AMBER      = (0, 176, 255)
AMBER_2    = (24, 122, 255)
AMBER_DIM  = (40, 120, 170)
GOOD       = (95, 232, 155)
BAD        = (71, 90, 255)

FONT  = cv2.FONT_HERSHEY_DUPLEX
FONTS = cv2.FONT_HERSHEY_SIMPLEX


# ── DRAW HELPERS ───────────────────────────────────────────── #
def rounded_rect(img, p1, p2, color, radius=8, thickness=-1):
    """Filled or outlined rectangle with rounded corners."""
    x1, y1 = p1
    x2, y2 = p2
    r = max(0, min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2))
    corners = [
        ((x1 + r, y1 + r), 180), ((x2 - r, y1 + r), 270),
        ((x2 - r, y2 - r), 0),   ((x1 + r, y2 - r), 90),
    ]
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
        for (cx, cy), a in corners:
            cv2.ellipse(img, (cx, cy), (r, r), 0, a, a + 90, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        for (cx, cy), a in corners:
            cv2.ellipse(img, (cx, cy), (r, r), 0, a, a + 90, color, thickness, cv2.LINE_AA)


def alpha_rect(img, p1, p2, color, alpha, radius=0):
    """Semi-transparent (optionally rounded) filled rectangle."""
    overlay = img.copy()
    if radius > 0:
        rounded_rect(overlay, p1, p2, color, radius, -1)
    else:
        cv2.rectangle(overlay, p1, p2, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def text(img, s, org, scale, color, thick=1, font=FONTS, spacing=0.0):
    """Anti-aliased text, optional letter spacing for tracked labels."""
    if spacing <= 0:
        cv2.putText(img, s, org, font, scale, color, thick, cv2.LINE_AA)
        return
    x, y = org
    for ch in s:
        cv2.putText(img, ch, (x, y), font, scale, color, thick, cv2.LINE_AA)
        (cw, _), _ = cv2.getTextSize(ch, font, scale, thick)
        x += cw + int(spacing)


def draw_crop_marks(frame, w, h, top=22):
    """Viewfinder corner brackets (top inset below the title bar)."""
    m, ln, col = 22, 26, CREAM
    overlay = frame.copy()
    for (cx, cy, dx, dy) in [(m, top, 1, 1), (w - m, top, -1, 1),
                             (m, h - m, 1, -1), (w - m, h - m, -1, -1)]:
        cv2.line(overlay, (cx, cy), (cx + dx * ln, cy), col, 2, cv2.LINE_AA)
        cv2.line(overlay, (cx, cy), (cx, cy + dy * ln), col, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)


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


def create_sidebar_panel(labels, predictions=None, width=340, height=600, active_idx=None):
    """Probability spectrum panel — ink background, amber accent."""
    sidebar = np.full((height, width, 3), INK_1[0], dtype=np.uint8)
    sidebar[:] = INK_1

    # Header
    pad = 22
    text(sidebar, "SPETTRO CONFIDENZA", (pad, 34), 0.5, CREAM, 1, spacing=2)
    text(sidebar, "ALFABETO LIS", (pad, 54), 0.36, CREAM_MUTE, 1, spacing=2)
    if predictions is not None:
        count = f"{len(labels)}"
        (cw, ch), _ = cv2.getTextSize(count, FONT, 0.8, 2)
        text(sidebar, count, (width - pad - cw, 50), 0.8, AMBER, 2, font=FONT)
    cv2.line(sidebar, (pad, 72), (width - pad, 72), LINE, 1, cv2.LINE_AA)

    # Rows
    top = 90
    item_h = (height - top - 16) // len(labels)
    bar_h = 7
    letter_x, bar_x = pad + 6, pad + 36
    bar_w = width - bar_x - 56

    for i, label in enumerate(labels):
        cy = top + i * item_h + item_h // 2
        is_active = (active_idx is not None and i == active_idx)
        prog = float(predictions[i]) if predictions is not None else 0.0

        # active row highlight
        if is_active:
            alpha_rect(sidebar, (pad - 6, cy - item_h // 2 + 1),
                       (width - pad + 6, cy + item_h // 2 - 1), AMBER, 0.14, radius=4)

        # letter
        l_col = AMBER if is_active else CREAM_MUTE
        text(sidebar, label.upper(), (letter_x, cy + 7), 0.62, l_col, 1, font=FONT)

        # track
        rounded_rect(sidebar, (bar_x, cy - bar_h // 2),
                     (bar_x + bar_w, cy + bar_h // 2), TRACK, bar_h // 2, -1)

        # fill
        if predictions is not None and prog > 0.01:
            fw = max(bar_h, int(bar_w * prog))
            fill_col = AMBER if prog > 0.7 else (AMBER_DIM if prog > 0.4 else CREAM_FAINT)
            rounded_rect(sidebar, (bar_x, cy - bar_h // 2),
                         (bar_x + fw, cy + bar_h // 2), fill_col, bar_h // 2, -1)

        # percentage
        pct = f"{int(prog * 100):>3d}%"
        p_col = CREAM if is_active else CREAM_FAINT
        text(sidebar, pct, (bar_x + bar_w + 10, cy + 5), 0.42, p_col, 1)

    return sidebar


def draw_prediction_overlay(frame, prediction, confidence, threshold, h, w):
    """Hero prediction overlay at the bottom of the video."""
    ph = 168
    py = h - ph
    # stacked alpha for a gradient-like fade
    alpha_rect(frame, (0, py), (w, h), INK, 0.55)
    alpha_rect(frame, (0, py + ph // 3), (w, h), INK, 0.55)
    cv2.line(frame, (0, py), (w, py), LINE, 1, cv2.LINE_AA)

    if prediction:
        if confidence > threshold:
            status, s_col = "RILEVATO", AMBER
        elif confidence > 0.3:
            status, s_col = "FORSE", AMBER_2
        else:
            status, s_col = "INCERTO", BAD
    else:
        status, s_col = "RICERCA", CREAM_MUTE

    # status row with leading dash
    sx, sy = 34, py + 38
    cv2.line(frame, (sx, sy - 5), (sx + 18, sy - 5), s_col, 1, cv2.LINE_AA)
    text(frame, status, (sx + 28, sy), 0.46, s_col, 1, spacing=3)

    if prediction:
        # hero letter
        letter = prediction.upper()
        text(frame, letter, (sx, h - 26), 4.6, AMBER, 6, font=FONT)
        (lw, _), _ = cv2.getTextSize(letter, FONT, 4.6, 6)

        # readout block to the right of the letter
        rx = sx + lw + 48
        text(frame, "CONFIDENZA", (rx, py + 78), 0.4, CREAM_MUTE, 1, spacing=2)
        text(frame, f"{int(confidence * 100)}%", (rx, py + 116), 1.2, CREAM, 2, font=FONT)
        # meter
        mw = max(180, w - rx - 60)
        my = py + 132
        rounded_rect(frame, (rx, my), (rx + mw, my + 6), TRACK, 3, -1)
        fw = int(mw * confidence)
        if fw > 6:
            rounded_rect(frame, (rx, my), (rx + fw, my + 6), AMBER, 3, -1)
    else:
        text(frame, "Mostra la mano destra alla telecamera.",
             (sx, py + 96), 0.8, CREAM_DIM, 1, font=FONT)


def draw_hand_detection_indicator(frame, has_hand, w, h):
    """Hand-detection tag (top-left, under the crop marks)."""
    label = "MANO RILEVATA" if has_hand else "IN ATTESA MANO"
    col = GOOD if has_hand else CREAM_MUTE
    x, y = 64, 64
    (tw, th), _ = cv2.getTextSize(label, FONTS, 0.42, 1)
    pad = 12
    x2 = x + pad + 16 + tw + pad
    y2 = y + 30
    alpha_rect(frame, (x, y), (x2, y2), INK, 0.6, radius=4)
    rounded_rect(frame, (x, y), (x2, y2), col if has_hand else LINE, 4, 1)
    cv2.circle(frame, (x + pad + 5, (y + y2) // 2), 4, col, -1, cv2.LINE_AA)
    text(frame, label, (x + pad + 16, (y + y2) // 2 + 4), 0.42, col, 1, spacing=1)


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
sidebar_width = 340
sidebar_height = 600

prev_t = time.time()
fps = 0.0

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
        active_idx = None

        # Check for RIGHT hand specifically
        has_right_hand = results.right_hand_landmarks is not None

        if has_right_hand:
            feats = np.array([points_detection(results)])
            prediction = model.predict(feats)[0]
            proba = model.predict_proba(feats)[0]
            pred_prob = float(np.max(proba))
            predictions_array = proba
            active_idx = int(np.argmax(proba))

        # Create sidebar panel
        sidebar = create_sidebar_panel(labels, predictions_array, sidebar_width,
                                       sidebar_height, active_idx)

        # Resize frame to match sidebar height if needed
        if h != sidebar_height:
            aspect_ratio = w / h
            new_width = int(sidebar_height * aspect_ratio)
            frame = cv2.resize(frame, (new_width, sidebar_height))
            h, w, c = frame.shape

        # FPS (smoothed)
        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Viewfinder crop marks (top inset below the title bar)
        draw_crop_marks(frame, w, h, top=60)

        # Top title bar
        alpha_rect(frame, (0, 0), (w, 48), INK, 0.62)
        cv2.line(frame, (0, 48), (w, 48), LINE, 1, cv2.LINE_AA)
        text(frame, "L", (18, 33), 0.95, AMBER, 2, font=FONT)
        text(frame, "IS", (34, 33), 0.95, CREAM, 2, font=FONT)
        text(frame, "RICONOSCIMENTO ALFABETO", (74, 31), 0.42, CREAM_DIM, 1, spacing=2)
        # live + fps (right side)
        live = "LIVE"
        (lw, _), _ = cv2.getTextSize(live, FONTS, 0.42, 1)
        fx = w - lw - 24
        cv2.circle(frame, (fx - 12, 24), 4, BAD, -1, cv2.LINE_AA)
        text(frame, live, (fx, 28), 0.42, CREAM, 1, spacing=2)
        text(frame, f"{fps:4.1f} FPS", (fx - 110, 28), 0.4, CREAM_MUTE, 1)

        # Overlays on main video
        draw_prediction_overlay(frame, prediction, pred_prob, args.threshold_prediction, h, w)
        draw_hand_detection_indicator(frame, has_right_hand, w, h)

        # Combine main video and sidebar horizontally (1px divider)
        divider = np.full((h, 1, 3), LINE[0], dtype=np.uint8)
        divider[:] = LINE
        combined_frame = np.hstack([frame, divider, sidebar])

        # Show combined window
        cv2.imshow('LIS: Sign Language Recognition System', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
