import cv2
import numpy as np
from django.shortcuts import render
from django.http import StreamingHttpResponse
from .hand_tracker import HandDetector

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Detector
detector = HandDetector()

# Drawing state
prev_x, prev_y = 0, 0
canvas = None

brush_color = (180, 0, 180)
brush_thickness = 6
eraser_mode = False

# 🎨 Colors
colors = [
    (255, 0, 255),   # pink
    (255, 0, 0),     # blue
    (0, 255, 0),     # green
    (0, 255, 255),   # yellow
    (0, 0, 255),     # red
]


def index(request):
    return render(request, "index.html")


def generate_frames():
    global prev_x, prev_y, canvas
    global brush_color, eraser_mode

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # Init canvas
        if canvas is None:
            canvas = np.zeros_like(frame)

        # Get frame size
        h, w, _ = frame.shape

        # ================= UI SETTINGS =================
        UI_HEIGHT = int(h * 0.12)
        BTN_WIDTH = int(w * 0.08)
        BTN_HEIGHT = int(UI_HEIGHT * 0.7)
        GAP = int(w * 0.02)

        START_X = int(w * 0.03)
        START_Y = int(UI_HEIGHT * 0.15)

        # ================= UI BAR =================
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, UI_HEIGHT), (255, 255, 255), -1)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        # ================= COLOR BUTTONS =================
        color_zones = []

        for i, color in enumerate(colors):
            x1 = START_X + i * (BTN_WIDTH + GAP)
            y1 = START_Y
            x2 = x1 + BTN_WIDTH
            y2 = y1 + BTN_HEIGHT

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

            color_zones.append((x1, y1, x2, y2))

            # Highlight selected
            if brush_color == color and not eraser_mode:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)

        # ================= ERASER =================
        ER_WIDTH = int(w * 0.12)

        ex1 = w - ER_WIDTH - GAP
        ey1 = START_Y
        ex2 = w - GAP
        ey2 = ey1 + BTN_HEIGHT

        eraser_zone = (ex1, ey1, ex2, ey2)

        cv2.putText(frame, "ERASER",
                    (ex1 + 10, ey2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)

        if eraser_mode:
            cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 255, 255), 3)

        # ================= HAND DETECTION =================
        frame = detector.find_hands(frame)
        lm_list = detector.find_position(frame)
        fingers = detector.fingers_up()

        if lm_list:
            x, y = lm_list[8][1], lm_list[8][2]

            # Pointer
            cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)

            # 🎯 COLOR SELECT
            for i, (x1, y1, x2, y2) in enumerate(color_zones):
                if x1 < x < x2 and y1 < y < y2:
                    brush_color = colors[i]
                    eraser_mode = False

            # 🧽 ERASER SELECT 
            ex1, ey1, ex2, ey2 = eraser_zone
            if ex1 < x < ex2 and ey1 < y < ey2:
                eraser_mode = True

            # ✍️ DRAW MODE (only index finger)
            if fingers[1] == 1 and sum(fingers) == 1:

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                if eraser_mode:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 25)
                else:
                    cv2.line(canvas, (prev_x, prev_y), (x, y),
                             brush_color, brush_thickness)

                prev_x, prev_y = x, y

            else:
                prev_x, prev_y = 0, 0

        # ================= MERGE =================
        frame = cv2.add(frame, canvas)

        # Encode
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def video_feed(request):
    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )