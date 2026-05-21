import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_this_dir))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64
import json

app = FastAPI(title="LIS Recognition API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_path = os.environ.get(
    "MODEL_PATH",
    os.path.join(_project_root, "models", "model_svm_all.sav"),
)
model = pickle.load(open(_model_path, "rb"))
labels: list[str] = [str(c) for c in model.classes_]

mp_holistic = mp.solutions.holistic


def _mediapipe_detection(frame, holistic):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = holistic.process(img)
    img.flags.writeable = True
    return results


def _points_detection(results):
    lm = results.right_hand_landmarks.landmark
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    rh = np.array([[p.x, p.y, p.z] for p in lm]).flatten()
    rh[0::3] = (rh[0::3] - x_min) / (x_max - x_min)
    rh[1::3] = (rh[1::3] - y_min) / (y_max - y_min)
    return rh


@app.get("/health")
async def health():
    return {"status": "ok", "labels": labels}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            try:
                data = await websocket.receive_text()

                if "," in data:
                    data = data.split(",", 1)[1]

                frame = cv2.imdecode(
                    np.frombuffer(base64.b64decode(data), np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if frame is None:
                    continue

                results = _mediapipe_detection(frame, holistic)
                has_hand = results.right_hand_landmarks is not None

                payload: dict = {
                    "has_hand": has_hand,
                    "prediction": None,
                    "confidence": 0.0,
                    "probabilities": {lbl: 0.0 for lbl in labels},
                }

                if has_hand:
                    features = np.array([_points_detection(results)])
                    pred = str(model.predict(features)[0])
                    proba = model.predict_proba(features)[0]
                    payload["prediction"] = pred
                    payload["confidence"] = float(np.max(proba))
                    payload["probabilities"] = {
                        str(lbl): float(p) for lbl, p in zip(labels, proba)
                    }

                await websocket.send_text(json.dumps(payload))

            except WebSocketDisconnect:
                break
            except Exception as exc:
                print(f"[ws] error: {exc}")
