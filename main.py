# === GestureX: MediaPipe Hands + Angle-based gesture detection with smoothing ===
# Control Keys:
#   q  -> quit
#   m  -> toggle mirror on/off
#
# In terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   venv\Scripts\activate

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import math
from collections import deque, Counter
import cv2
import mediapipe as mp

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
    EXTEND_TH = 160
    FOLD_TH   = 145

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
      - If ambiguous, prefer last_label over 'Other'
    """
    up, folded, angles = finger_states_angle(lm)
    fingers_up = sum(up.values())
    fingers_folded = sum(folded.values())
    four_fingers_up = up["index"] and up["middle"] and up["ring"] and up["pinky"]
    four_fingers_folded = folded["index"] and folded["middle"] and folded["ring"] and folded["pinky"]

    if four_fingers_folded and not up["thumb"]:
        return "Closed Fist"
    if four_fingers_up and not folded["thumb"]:
        return "Open Palm"
    if up["index"] and not up["middle"] and not up["ring"] and not up["pinky"] and not up["thumb"]:
        return "Point (Index)"
    if not up["index"] and up["middle"] and not up["ring"] and not up["pinky"] and not up["thumb"]:
        return "Not Nice!" # fun idea to blur middle finger
    if up["index"] and up["middle"] and not up["ring"] and not up["pinky"] and not up["thumb"]:
        return "Peace / V"
    if up["thumb"] and not up["index"] and not up["middle"] and not up["ring"] and not up["pinky"]:
        return "Thumbs Up/Side"
    if up["index"] and up["pinky"] and not up["middle"] and not up["ring"]:
        return "Rock 🤘"
    return "Other"
    # maybe implent pinch for volume control (thumb and finger close => lower, farther apart => higher)

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
            votes = self.min_agree  # fake stability to keep current

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
        raise RuntimeError("Could not open camera. Try a different index (1, 2, ...) or check Windows Camera app.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smoother = StablePrinter(window=9, min_agree=6, allow_other_after=12)
    mirror = True

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
            frame, "[q]=quit  [m]=mirror", (20, frame.shape[0]-15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("GestureX", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mirror = not mirror

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()