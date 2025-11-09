# === GestureX: MediaPipe Hands + Angle-based gesture detection with smoothing ===
# Control Keys:
#   q  -> quit
#   m  -> toggle mirror on/off
#
# In terminal:
#   python -m venv venv
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   venv\Scripts\activate
#   pip install opencv-python mediapipe

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import math
from collections import deque, Counter
import cv2
import mediapipe as mp
import numpy as np

def _lm_np(lm_pt):
    return np.array([lm_pt.x, lm_pt.y, lm_pt.z], dtype=np.float32)

def _unit(v):
    n = np.linalg.norm(v) + 1e-9
    return v / n

def hand_plane_axis(lm):
    wrist = _lm_np(lm[0])
    index = _lm_np(lm[5])
    pinky = _lm_np(lm[17])

    ex = _unit(index - wrist)
    ey_raw = pinky - wrist
    ey = _unit(ey_raw - np.dot(ey_raw, ex) * ex)
    return wrist, ex, ey

def proj_plane_uv(pt, origin, ex, ey):
    v = _lm_np(pt) - origin
    u = float(np.dot(v, ex))
    v2 = float(np.dot(v, ey))
    return u, v2

def _angle(a, b, c):
    """
    Returns the angle (degrees) at point b formed by points a-b-c.
    Uses MediaPipe normalized landmark coordinates (x,y in [0,1]).
    """
    bax, bay = a.x - b.x, a.y - b.y
    bcx, bcy = c.x - b.x, c.y - b.y
    dot = bax * bcx + bay * bcy
    n1 = math.hypot(bax, bay)
    n2 = math.hypot(bcx, bcy)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cosang))

def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def is_pinch(lm, last_label=None):
        ti = _dist(lm[4], lm[8])
        pw = _dist(lm[5], lm[17])

        ENTER, EXIT = 0.14, 0.22

        if last_label == "Pinch":
            return ti <= EXIT * pw
        else:
            return ti <= ENTER * pw

def finger_states_angle(lm):
    """
    Angle-based extension detection for all 5 fingers.
    Returns:
      up     -> dict of booleans {thumb/index/middle/ring/pinky} True if extended/straight
      folded -> dict of booleans True if clearly bent/folded
    Notes:
      - Uses angle thresholds at key joints.
      - Uses a 'dead zone' between thresholds to reduce jitter.
      - Angle >= EXTEND_TH => straight/extended; <= FOLD_TH => bent/folded.
    """
    EXTEND_TH = 165
    FOLD_TH   = 140

    thumb_ang  = _angle(lm[2], lm[3], lm[4])
    index_ang  = _angle(lm[5],  lm[6],  lm[8])
    middle_ang = _angle(lm[9],  lm[10], lm[12])
    ring_ang   = _angle(lm[13], lm[14], lm[16])
    pinky_ang  = _angle(lm[17], lm[18], lm[20])

    angles = {
        "thumb":  thumb_ang,
        "index":  index_ang,
        "middle": middle_ang,
        "ring":   ring_ang,
        "pinky":  pinky_ang,
    }

    up = {k: (v >= EXTEND_TH) for k, v in angles.items()}
    folded = {k: (v <= FOLD_TH) for k, v in angles.items()}
    return up, folded, angles

def classify_gesture(lm, last_label=None):
    """
    Classifies a few common gestures using angle-based states.
    Returns a string label.
    Notes:
      - "Open Palm" allows slightly bent thumb
      - "Closed Fist" allows slight bends
      - If ambiguous returns 'Other'
    """
    up, folded, angles = finger_states_angle(lm)

    if up["thumb"] and folded["index"] and folded["middle"] and folded["ring"] and folded["pinky"]:
      return "Thumbs Up/Side"
    if up["index"] and folded["middle"] and folded["ring"] and folded["pinky"] and (not up["thumb"]):
        return "Point (Index)"
    if up["index"] and up["middle"] and folded["ring"] and folded["pinky"] and (not up["thumb"]):
       return "Peace / V"
    if up["index"] and up["pinky"] and folded["middle"] and folded["ring"] and (not up["thumb"]):
       return "Rock On!"
    if (not up["thumb"]) and folded["index"] and up["middle"] and folded["ring"] and folded["pinky"]:
        return "WOAH!"
    if up["index"] and up["middle"] and up["ring"] and up["pinky"] and (not folded["thumb"]):
       return "Open Palm"
    if folded["index"] and folded["middle"] and folded["ring"] and folded["pinky"] and (not up["thumb"]):
      return "Closed Fist"
    return "Other"


class StablePrinter:
    """
    Smooths fast-changing labels:
      - keeps a rolling window of last N labels
      - prints (and updates on-screen) only when a majority agrees AND it differs from the last shown label
    """
    def __init__(self, window=9, min_agree=6, allow_other_after=12, on_change_print=True):
        self.history = deque(maxlen=window)
        self.current = None
        self.min_agree = min_agree
        self.on_change_print = on_change_print
        self.other_streak = 0
        self.allow_other_after = allow_other_after

    def update(self, label):
        if label == "Other":
            self.other_streak += 1
        else:
            self.other_streak = 0

        self.history.append(label)
        counts = Counter(self.history)
        top_label, votes = counts.most_common(1)[0]

        if top_label == "Other" and self.other_streak < self.allow_other_after:
            top_label = self.current or "Other"
            votes = self.min_agree

        if votes >= self.min_agree and top_label != self.current:
            self.current = top_label
            if self.on_change_print:
                print(self.current)

        return self.current

def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    smoother = StablePrinter(window=9, min_agree=6, allow_other_after=12)
    mirror = True
    pinch = False

    while True:
        success, frame = cap.read()
        if not success:
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if pinch:
                    origin, ex, ey = hand_plane_axis(hand_landmarks.landmark)

                    u4, v4 = proj_plane_uv(hand_landmarks.landmark[4], origin, ex, ey)
                    u8, v8 = proj_plane_uv(hand_landmarks.landmark[8], origin, ex, ey)
                    ti_plane = float(np.hypot(u8 - u4, v8 - v4))

                    ui, vi = proj_plane_uv(hand_landmarks.landmark[5], origin, ex, ey)
                    u8b, v8b = proj_plane_uv(hand_landmarks.landmark[8], origin, ex, ey)
                    index_len = float(np.hypot(u8b - ui, v8b - vi))

                    ut2, vt2 = proj_plane_uv(hand_landmarks.landmark[2], origin, ex, ey)
                    ut4, vt4 = proj_plane_uv(hand_landmarks.landmark[4], origin, ex, ey)
                    thumb_len = float(np.hypot(ut4 - ut2, vt4 - vt2))

                    baseline = max(1e-6, 0.5 * (index_len + thumb_len))
                    ratio = ti_plane / baseline

                    CLOSE, FAR = 0.25, 1.50
                    val = (ratio - CLOSE) / (FAR - CLOSE)
                    val = 0.0 if val < 0 else 1.0 if val > 1 else val
                    open_pct = int(round(val * 100))

                    h, w, _ = frame.shape
                    tt = hand_landmarks.landmark[4]; it = hand_landmarks.landmark[8]
                    x1, y1 = int(tt.x * w), int(tt.y * h)
                    x2, y2 = int(it.x * w), int(it.y * h)

                    r = int(255 * val)
                    g = int(255 * (1 - val))
                    cv2.line(frame, (x1, y1), (x2, y2), (0, g, r), 4)

                    cv2.putText(frame, f"Open (plane): {open_pct}%", (20, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                

                # thumb_tip = hand_landmarks.landmark[4]
                # index_tip = hand_landmarks.landmark[8]

                # h, w, _ = frame.shape
                # x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
                # x2, y2 = int(index_tip.x * w), int(index_tip.y * h)
                # distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

                # min_d, max_d = 20, 350
                # pinch_value = (distance - min_d) / (max_d - min_d)
                # pinch_value = max(0, min(1, pinch_value))
                # pinch_percent = int(pinch_value * 100)

                # r = int(255 * pinch_value)
                # g = int(255 * (1 - pinch_value))

                # cv2.line(frame, (x1, y1), (x2, y2), (0, g, r), 4)
                # cv2.putText(frame, f"Pinch: {pinch_percent}%", (20, 80),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                '''
                # Displays landmark numbers
                for i, lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.putText(frame, str(i), (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                '''

                last = smoother.current
                label = classify_gesture(hand_landmarks.landmark, last_label=last)
                smoother.update(label)
        else:
            smoother.update("Other")

        if smoother.current:
            cv2.putText(
                frame, smoother.current, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2, cv2.LINE_AA)

        cv2.putText(
            frame, "[q]=quit  [m]=mirror  [p]=toggle pinch", (20, frame.shape[0]-15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("GestureX", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mirror = not mirror
        elif key == ord('p'):
            pinch = not pinch

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()