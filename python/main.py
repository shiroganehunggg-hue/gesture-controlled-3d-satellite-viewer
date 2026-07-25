import cv2
import mediapipe as mp
import time

# ==== Init MediaPipe ====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils


def open_camera(start_index=0, max_index=4):
    """Try to open a webcam device using multiple device indexes and backends."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for device in range(start_index, start_index + max_index + 1):
        for backend in backends:
            cap = cv2.VideoCapture(device, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"Opened webcam device {device} with backend {backend}.")
                return cap
            cap.release()
    return None

HOLD_TIME_ROOT = 0.5
HOLD_TIME_SUB = 0.8
OVERLAY_ALPHA = 0.35

hover_zone = None
hover_start_time = None
selected_model = None
current_menu = "root"


def draw_centered_text(frame, text, x1, y1, x2, y2, color=(255, 255, 255),
                        font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.55, thickness=2):
    """Draw text centered inside a box (x1,y1)-(x2,y2)."""
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    cx = x1 + (x2 - x1 - text_w) // 2
    cy = y1 + (y2 - y1 + text_h) // 2
    cv2.putText(frame, text, (cx, cy), font, scale, color, thickness)


cap = open_camera()

if cap is None or not cap.isOpened():
    print(
        "Cannot open webcam.\n"
        "Try closing other camera apps, check camera permissions, or use a different device index."
    )
    exit()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Cannot read frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    ROOT_BOX = {"name": "Models", "rect": (w - 70, 20, w - 20, 60)}

    SUBMENU_BOXES = [
        {"name": "Satellite A", "rect": (w - 200, 30,  w - 20, 130)},
        {"name": "Satellite B", "rect": (w - 200, 150, w - 20, 250)},
        {"name": "ISS Station", "rect": (w - 200, 270, w - 20, 370)},
        {"name": "Back",        "rect": (w - 200, 390, w - 20, 460)},
    ]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    index_x, index_y = None, None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[8]
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            index_x, index_y = ix, iy

            cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)

    active_boxes = [ROOT_BOX] if current_menu == "root" else SUBMENU_BOXES
    hold_time = HOLD_TIME_ROOT if current_menu == "root" else HOLD_TIME_SUB

    current_hover = None
    overlay = frame.copy()

    for zone in active_boxes:
        x1, y1, x2, y2 = zone["rect"]
        inside = (
            index_x is not None and
            x1 <= index_x <= x2 and
            y1 <= index_y <= y2
        )

        if inside:
            current_hover = zone["name"]

        fill_color = (0, 200, 255) if inside else (60, 60, 60)

        if current_menu == "submenu":
            cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)

        border_color = (0, 255, 255) if inside else (200, 200, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)
        draw_centered_text(frame, zone["name"], x1, y1, x2, y2)

    if current_menu == "submenu":
        frame = cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0)

    # ==== Selection logic: hover only, no pinch needed ====
    if current_hover is not None:
        if hover_zone != current_hover:
            hover_zone = current_hover
            hover_start_time = time.time()
        else:
            elapsed = time.time() - hover_start_time
            progress = min(elapsed / hold_time, 1.0)

            zx1, zy1, zx2, zy2 = next(z["rect"] for z in active_boxes if z["name"] == current_hover)
            bar_width = int((zx2 - zx1) * progress)
            cv2.rectangle(frame, (zx1, zy2 - 8), (zx1 + bar_width, zy2), (0, 255, 0), -1)

            if progress >= 1.0:
                if current_menu == "root":
                    current_menu = "submenu"
                else:
                    if current_hover == "Back":
                        current_menu = "root"
                    else:
                        selected_model = current_hover

                hover_zone = None
                hover_start_time = None
    else:
        hover_zone = None
        hover_start_time = None

    if selected_model:
        cv2.putText(frame, f"Selected: {selected_model}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Hand Gesture Model Selector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
