import cv2
import mediapipe as mp
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

# MediaPipe hand setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Drawing canvas
canvas = None

# Previous drawing point
prev_x = None
prev_y = None

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to read from webcam")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Create canvas once
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Important landmarks
            index_tip = hand_landmarks.landmark[8]
            index_pip = hand_landmarks.landmark[6]

            middle_tip = hand_landmarks.landmark[12]
            middle_pip = hand_landmarks.landmark[10]

            ring_tip = hand_landmarks.landmark[16]
            ring_pip = hand_landmarks.landmark[14]

            pinky_tip = hand_landmarks.landmark[20]
            pinky_pip = hand_landmarks.landmark[18]

            # Index fingertip position
            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)

            # Finger up/down detection
            index_up = index_tip.y < index_pip.y
            middle_up = middle_tip.y < middle_pip.y
            ring_up = ring_tip.y < ring_pip.y
            pinky_up = pinky_tip.y < pinky_pip.y

            # ---------------- DRAW MODE ----------------
            if index_up and not middle_up:
                if prev_x is None or prev_y is None:
                    prev_x, prev_y = index_x, index_y

                cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (0, 255, 0), 5)
                prev_x, prev_y = index_x, index_y

                cv2.putText(frame, "DRAW MODE", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # ---------------- SELECTION MODE ----------------
            elif index_up and middle_up and not ring_up and not pinky_up:
                prev_x, prev_y = None, None

                cv2.putText(frame, "SELECTION MODE", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # ---------------- ERASER MODE ----------------
            elif not index_up and not middle_up and not ring_up and not pinky_up:
                prev_x, prev_y = None, None

                cv2.circle(canvas, (index_x, index_y), 30, (0, 0, 0), -1)

                cv2.putText(frame, "ERASER MODE", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.circle(frame, (index_x, index_y), 30, (0, 0, 255), 2)

            # ---------------- OTHER ----------------
            else:
                prev_x, prev_y = None, None

            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)

    else:
        prev_x, prev_y = None, None

    # Combine webcam + drawing canvas
    combined = cv2.add(frame, canvas)

    cv2.imshow("Hand Drawing System", combined)

    key = cv2.waitKey(1) & 0xFF

    # ESC to exit
    if key == 27:
        break

    # Press C to clear canvas
    if key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()