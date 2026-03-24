"""
Real-Time Gesture-to-Action Virtual Mouse
==========================================
Compatible with mediapipe 0.10.30+  (new Tasks API)

Dependencies:
    pip install opencv-python mediapipe pyautogui numpy

Gesture legend:
    - Middle finger curled DOWN  : tracking ACTIVE
    - Index fingertip            : moves the cursor
    - Index + Thumb pinch        : left click
    - Tighter pinch              : double-click
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions
import pyautogui
import numpy as np
import time
import urllib.request
import os
from collections import deque


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Config:
    CAMERA_INDEX          = 0
    FRAME_WIDTH           = 640
    FRAME_HEIGHT          = 480
    SMOOTH_BUFFER         = 6       # moving-average window size
    DEAD_ZONE_PX          = 5       # minimum pixel delta before cursor moves
    PINCH_CLICK_THRESHOLD = 0.045   # normalised pinch distance → click
    PINCH_DOUBLE_THRESH   = 0.025   # tighter pinch → double-click
    CLICK_HOLD_SECONDS    = 0.08    # hold time before click fires
    CLICK_DEBOUNCE        = 0.4     # minimum gap between clicks
    SCREEN_REGION         = None    # (left, top, right, bottom) or None
    FLIP_HORIZONTAL       = True
    SHOW_PREVIEW          = True
    USE_CLAHE             = True
    CLAHE_CLIP_LIMIT      = 2.0
    CLAHE_TILE_GRID       = (8, 8)
    MODEL_PATH            = "hand_landmarker.task"  # downloaded automatically


# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------
LM_THUMB_TIP    = 4
LM_INDEX_TIP    = 8
LM_MIDDLE_MCP   = 9
LM_MIDDLE_TIP   = 12


# ---------------------------------------------------------------------------
# Download the model file if not present
# ---------------------------------------------------------------------------
def ensure_model(path: str):
    if os.path.exists(path):
        return
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )
    print(f"[VirtualMouse] Downloading hand landmarker model -> {path} ...")
    urllib.request.urlretrieve(url, path)
    print("[VirtualMouse] Model downloaded.")


# ---------------------------------------------------------------------------
# Preprocessing: CLAHE for lighting robustness
# ---------------------------------------------------------------------------
def preprocess_frame(frame, clahe):
    if clahe is None:
        return frame
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ch = list(cv2.split(ycrcb))
    ch[0] = clahe.apply(ch[0])
    return cv2.cvtColor(cv2.merge(ch), cv2.COLOR_YCrCb2BGR)


# ---------------------------------------------------------------------------
# Moving-average smoothing filter
# ---------------------------------------------------------------------------
class PositionFilter:
    def __init__(self, size=6):
        self._x = deque(maxlen=size)
        self._y = deque(maxlen=size)

    def update(self, x, y):
        self._x.append(x)
        self._y.append(y)
        return float(np.mean(self._x)), float(np.mean(self._y))

    def reset(self):
        self._x.clear()
        self._y.clear()


# ---------------------------------------------------------------------------
# Gesture helpers
# ---------------------------------------------------------------------------
def is_tracking_active(landmarks) -> bool:
    """Middle finger tip below its base knuckle (y increases downward) = curled."""
    return landmarks[LM_MIDDLE_TIP].y > landmarks[LM_MIDDLE_MCP].y


def pinch_distance(landmarks, frame_w: int) -> float:
    ix = landmarks[LM_INDEX_TIP].x * frame_w
    iy = landmarks[LM_INDEX_TIP].y * frame_w
    tx = landmarks[LM_THUMB_TIP].x * frame_w
    ty = landmarks[LM_THUMB_TIP].y * frame_w
    return np.hypot(ix - tx, iy - ty) / frame_w


def map_to_screen(nx, ny, sw, sh, region=None):
    if region:
        l, t, r, b = region
        x = int(l + nx * (r - l))
        y = int(t + ny * (b - t))
    else:
        x = int(nx * sw)
        y = int(ny * sh)
    return max(0, min(x, sw - 1)), max(0, min(y, sh - 1))


# ---------------------------------------------------------------------------
# HUD overlay
# ---------------------------------------------------------------------------
def draw_hud(frame, tracking, pinch, pos, cfg):
    clr = (0, 220, 100) if tracking else (60, 60, 200)
    label = "TRACKING ON" if tracking else "TRACKING OFF  (curl middle finger)"
    cv2.putText(frame, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, clr, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Pinch: {pinch:.3f}  thresh: {cfg.PINCH_CLICK_THRESHOLD:.3f}",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    if pos:
        cv2.putText(frame, f"Cursor: {pos[0]}, {pos[1]}",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    bar = int(min(pinch / cfg.PINCH_CLICK_THRESHOLD, 1.0) * 200)
    cv2.rectangle(frame, (10, 92), (210, 104), (80, 80, 80), -1)
    bc = (0, 200, 80) if pinch < cfg.PINCH_CLICK_THRESHOLD else (0, 80, 220)
    cv2.rectangle(frame, (10, 92), (10 + bar, 104), bc, -1)


# ---------------------------------------------------------------------------
# Draw landmarks manually (compatible with new Tasks API)
# ---------------------------------------------------------------------------
def draw_landmarks(frame, landmarks, frame_w, frame_h):
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),(0,17),
    ]
    pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 120), 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        cv2.circle(frame, pt, 4, (0, 150, 80), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = Config()
    pyautogui.PAUSE = 0.0
    pyautogui.FAILSAFE = True
    screen_w, screen_h = pyautogui.size()

    ensure_model(cfg.MODEL_PATH)

    cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {cfg.CAMERA_INDEX}")

    clahe = (
        cv2.createCLAHE(clipLimit=cfg.CLAHE_CLIP_LIMIT,
                         tileGridSize=cfg.CLAHE_TILE_GRID)
        if cfg.USE_CLAHE else None
    )

    base_options = mp_python.BaseOptions(model_asset_path=cfg.MODEL_PATH)
    options = HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    pos_filter   = PositionFilter(cfg.SMOOTH_BUFFER)
    prev_raw     = None
    pinch_start  = None
    last_click_t = 0.0
    click_fired  = False

    print("[VirtualMouse] Running -- press Q in the preview window to quit.")
    print("[VirtualMouse] Curl your middle finger DOWN to activate tracking.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            if cfg.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            fh, fw = frame.shape[:2]
            processed = preprocess_frame(frame, clahe)
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            tracking = False
            pinch    = 1.0
            smoothed = None

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]

                if cfg.SHOW_PREVIEW:
                    draw_landmarks(frame, lm, fw, fh)

                tracking = is_tracking_active(lm)

                if tracking:
                    raw_x = lm[LM_INDEX_TIP].x
                    raw_y = lm[LM_INDEX_TIP].y

                    move = True
                    if prev_raw is not None:
                        dx = abs(raw_x - prev_raw[0]) * fw
                        dy = abs(raw_y - prev_raw[1]) * fh
                        if dx < cfg.DEAD_ZONE_PX and dy < cfg.DEAD_ZONE_PX:
                            move = False

                    if move:
                        prev_raw = (raw_x, raw_y)
                        sx, sy = pos_filter.update(raw_x, raw_y)
                        scr_x, scr_y = map_to_screen(sx, sy, screen_w, screen_h,
                                                      cfg.SCREEN_REGION)
                        smoothed = (scr_x, scr_y)
                        pyautogui.moveTo(scr_x, scr_y)

                    pinch = pinch_distance(lm, fw)
                    now   = time.monotonic()

                    if pinch < cfg.PINCH_CLICK_THRESHOLD:
                        if pinch_start is None:
                            pinch_start = now
                        hold = now - pinch_start
                        if (hold >= cfg.CLICK_HOLD_SECONDS
                                and not click_fired
                                and (now - last_click_t) > cfg.CLICK_DEBOUNCE):
                            if pinch < cfg.PINCH_DOUBLE_THRESH:
                                pyautogui.doubleClick()
                                print("[VirtualMouse] Double-click")
                            else:
                                pyautogui.click()
                                print("[VirtualMouse] Click")
                            last_click_t = now
                            click_fired  = True
                    else:
                        pinch_start = None
                        click_fired = False

                else:
                    pos_filter.reset()
                    prev_raw = None

            if cfg.SHOW_PREVIEW:
                draw_hud(frame, tracking, pinch, smoothed, cfg)
                cv2.imshow("VirtualMouse  --  press Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print("[VirtualMouse] Exited cleanly.")


if __name__ == "__main__":
    main()
