import cv2
import mediapipe as mp
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Can't open camera")
    exit()

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,             # support both hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Drawing canvas
canvas = None

# Previous drawing point
prev_x = None
prev_y = None

# Smoothed fingertip position (for smoother lines)
smooth_x = None
smooth_y = None
smooth_factor = 0.15
min_move = 3

# Brush settings
brush_thickness = 5
eraser_thickness = 30

# Color palette (modern, softer colors)
colors = {
    'green':  (0, 220, 0),
    'red':    (0, 0, 220),
    'blue':   (220, 0, 0),
    'yellow': (0, 220, 220),
    'eraser': (0, 0, 0),
    'clear':  (0, 0, 0),  # special action, not color
}
color_names = ['green', 'red', 'blue', 'yellow', 'eraser', 'clear']
selected_color = 'green'  # default

# Font / style
font = cv2.FONT_HERSHEY_SIMPLEX

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

    # --- MODERN TOP TOOLBAR ---
    bar_height = 70
    cv2.rectangle(frame, (0, 0), (w, bar_height), (40, 40, 40), -1)  # dark bar

    # Button width
    btn_w = 90
    btn_pad = 10
    total_width = btn_pad + 6 * (btn_w + btn_pad)

    # Center buttons
    start_x = (w - total_width) // 2 if w > total_width else btn_pad

    button_rects = []
    for i, cname in enumerate(color_names):
        x = start_x + i * (btn_w + btn_pad)
        y = 10
        rect = (x, y, x + btn_w, y + bar_height - 20)
        button_rects.append((rect, cname))

        # Background color
        if cname == 'eraser':
            cv2.rectangle(frame, rect[:2], rect[2:], (80, 80, 80), -1)
        elif cname == 'clear':
            cv2.rectangle(frame, rect[:2], rect[2:], (100, 100, 100), -1)
        else:
            cv2.rectangle(frame, rect[:2], rect[2:], colors[cname], -1)

        # Border if active
        if cname == selected_color:
            cv2.rectangle(frame, rect[:2], rect[2:], (255, 255, 255), 2)

        # Label
        text = cname.upper() if cname != 'eraser' else 'ERASER'
        text_size = cv2.getTextSize(text, font, 0.55, 2)[0]
        tx = x + (btn_w - text_size[0]) // 2
        ty = y + (bar_height - 20 + text_size[1]) // 2
        cv2.putText(frame, text, (tx, ty), font, 0.55,
                    (255, 255, 255), 1 if cname != 'green' else 2)

    # --- HAND DETECTION ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    mode = "NONE"
    right_index_x, right_index_y = None, None
    left_thumb_up = False          # control: drawing allowed?

    if results.multi_hand_landmarks and results.multi_handedness:
        for i_hand, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get handedness label
            handedness = results.multi_handedness[i_hand]
            hand_label = handedness.classification[0].label.lower()  # "left" or "right"

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(100, 220, 255), thickness=2),
                mp_draw.DrawingSpec(color=(100, 150, 255), thickness=2)
            )

            # === LEFT HAND: thumb up/down control ===
            if hand_label == "left":
                thumb_tip = hand_landmarks.landmark[4]
                thumb_mcp = hand_landmarks.landmark[2]

                # Thumb up detection (for left hand, thumb pointing right)
                thumb_x = thumb_tip.x
                thumb_mcp_x = thumb_mcp.x
                left_thumb_up = thumb_x > thumb_mcp_x   # left thumb up to the right

                # Visual marker for left thumb
                lx = int(thumb_tip.x * w)
                ly = int(thumb_tip.y * h)
                cv2.circle(frame, (lx, ly), 8, (255, 255, 255), -1)
                cv2.circle(frame, (lx, ly), 10, (0, 255, 255), 2)

            # === RIGHT HAND: drawing with index finger ===
            elif hand_label == "right":
                index_tip   = hand_landmarks.landmark[8]
                index_pip   = hand_landmarks.landmark[6]
                middle_tip  = hand_landmarks.landmark[12]
                middle_pip  = hand_landmarks.landmark[10]
                ring_tip    = hand_landmarks.landmark[16]
                ring_pip    = hand_landmarks.landmark[14]
                pinky_tip   = hand_landmarks.landmark[20]
                pinky_pip   = hand_landmarks.landmark[18]

                ix = int(index_tip.x * w)
                iy = int(index_tip.y * h)

                # Finger up/down
                index_up   = index_tip.y < index_pip.y
                middle_up  = middle_tip.y < middle_pip.y

                # Smooth fingertip tracking (only for right hand)
                if smooth_x is None or smooth_y is None:
                    smooth_x, smooth_y = ix, iy

                dx = ix - smooth_x
                dy = iy - smooth_y
                dist = np.sqrt(dx ** 2 + dy ** 2)

                if dist > min_move:
                    smooth_x += int(dx * smooth_factor)
                    smooth_y += int(dy * smooth_factor)

                right_index_x, right_index_y = smooth_x, smooth_y

                # --- SELECTION MODE: index + middle up (right hand) ---
                if index_up and middle_up:
                    mode = "SELECT"
                    prev_x, prev_y = None, None

                    if right_index_y < bar_height:
                        for (rect, cname) in button_rects:
                            rx0, ry0, rx1, ry1 = rect
                            if rx0 <= right_index_x <= rx1 and ry0 <= right_index_y <= ry1:
                                if cname == "clear":
                                    canvas = np.zeros((h, w, 3), dtype=np.uint8)
                                else:
                                    selected_color = cname

                # Drawing only if:
                # - right index is up
                # - left thumb is DOWN
                elif index_up and not left_thumb_up:
                    mode = "DRAW"

                    if prev_x is None or prev_y is None:
                        prev_x, prev_y = right_index_x, right_index_y

                    if selected_color == "eraser":
                        cv2.line(canvas, (prev_x, prev_y), (right_index_x, right_index_y),
                                 colors[selected_color], eraser_thickness)
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (right_index_x, right_index_y),
                                 colors[selected_color], brush_thickness)

                    prev_x, prev_y = right_index_x, right_index_y

                else:
                    mode = "STOPPED"
                    prev_x, prev_y = None, None

                # Draw fingertip marker for right index
                cv2.circle(frame, (right_index_x, right_index_y), 8,
                           (255, 255, 255), -1)
                cv2.circle(frame, (right_index_x, right_index_y), 10,
                           (0, 150, 255), 2)

    # If no hand is detected, drawing is disabled
    if results.multi_hand_landmarks is None:
        mode = "STOPPED"
        prev_x, prev_y = None, None

    # --- STATUS TEXT ---
    if mode == "DRAW":
        cv2.putText(frame, "DRAW", (10, h - 40), font, 1, (0, 255, 0), 2)
    elif mode == "SELECT":
        cv2.putText(frame, "SELECT", (10, h - 40), font, 1, (255, 255, 0), 2)
    elif mode == "STOPPED":
        cv2.putText(frame, "STOP", (10, h - 40), font, 1, (0, 0, 255), 2)

    # --- COMBINE FRAME + CANVAS ---
    combined = cv2.add(frame, canvas)

    # Resize window for a bit more modern look
    cv2.namedWindow("Hand Drawing", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Drawing", 1024, 720)
    cv2.imshow("Hand Drawing", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()