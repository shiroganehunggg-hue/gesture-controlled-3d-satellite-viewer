import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils


def open_camera(start_index=0, max_index=4):
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

FIST_THRESHOLD = 1.3  # tuỳ chỉnh độ nhạy, thử nghiệm để tìm số phù hợp

def distance(a, b, w, h):
    ax, ay = a.x * w, a.y * h
    bx, by = b.x * w, b.y * h
    return math.hypot(ax - bx, ay - by)

def is_fist(landmarks, w, h):
    wrist = landmarks[0]
    middle_mcp = landmarks[9]  # gốc ngón giữa, dùng làm thước đo kích thước bàn tay
    palm_size = distance(wrist, middle_mcp, w, h)

    fingertip_ids = [8, 12, 16, 20]  # trỏ, giữa, áp út, út
    total_distance = 0
    for tip_id in fingertip_ids:
        tip = landmarks[tip_id]
        total_distance += distance(wrist, tip, w, h)

    avg_distance = total_distance / len(fingertip_ids)
    ratio = avg_distance / palm_size

    return ratio < FIST_THRESHOLD, ratio


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
        print("Cannot read frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_text = "No hand"
    gesture_color = (200, 200, 200)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fist, ratio = is_fist(hand_landmarks.landmark, w, h)

            if fist:
                gesture_text = f"GRAB (fist)  ratio={ratio:.2f}"
                gesture_color = (0, 0, 255)
            else:
                gesture_text = f"OPEN (release)  ratio={ratio:.2f}"
                gesture_color = (0, 255, 0)

    cv2.putText(frame, gesture_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)

    cv2.imshow("Gesture Test - Fist vs Open", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
