Gesture Virtual Mouse
A fun computer vision project that lets you control your mouse using just your hand and a webcam. No extra hardware needed.
I built this using Python, OpenCV, and Mediapipe for hand tracking, and PyAutoGUI to actually move the cursor around the screen.

How it works
Your webcam picks up your hand and tracks 21 points on it in real time. The tip of your index finger controls where the cursor goes, and pinching your index finger and thumb together acts as a click.
To avoid the cursor going crazy whenever your hand is in frame, there's a toggle — curl your middle finger down to turn tracking on, extend it to turn it off.

Gestures

Middle finger curled down — tracking on
Middle finger up — tracking off
Move index finger — moves the cursor
Pinch index + thumb — left click
Tight pinch — double click


Setup
Make sure you have Python 3.10 or 3.11 installed, then run:
bashpython -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
python virtual_mouse.py
The first time you run it, it'll download a small model file (~25 MB) automatically. After that it starts up instantly.

Things you can tweak
Everything is in the Config class at the top of virtual_mouse.py. The main ones worth adjusting:

SMOOTH_BUFFER — how smooth the cursor feels (higher = smoother but a bit laggy)
DEAD_ZONE_PX — how much your hand can shake before the cursor reacts
PINCH_CLICK_THRESHOLD — how close your fingers need to be to trigger a click
CAMERA_INDEX — change to 1 or 2 if it opens the wrong camera


Common issues
Wrong camera opens — change CAMERA_INDEX in the config
Cursor is too jittery — increase SMOOTH_BUFFER to something like 10
Keeps clicking by accident — increase CLICK_HOLD_SECONDS to 0.15 or so
Bad detection in low light — USE_CLAHE is already on by default, but try adding a lamp facing your hand
macOS mouse control not working — go to System Settings → Privacy & Security → Accessibility and give your terminal permission
