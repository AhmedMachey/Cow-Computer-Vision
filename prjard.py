import cv2
import mediapipe as mp
import serial
import time
import math

# Initialize serial communication
arduino = serial.Serial(port='COM5', baudrate=9600, timeout=1)
time.sleep(2)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Compute distance between thumb and index
def thumb_index_distance(hand_landmarks, image_w, image_h):
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]

    x1, y1 = int(thumb.x * image_w), int(thumb.y * image_h)
    x2, y2 = int(index.x * image_w), int(index.y * image_h)

    dist = math.hypot(x2 - x1, y2 - y1)
    return dist, (x1, y1, x2, y2)

# Determine left/right hand
def is_left(hand):  return hand.classification[0].label == "Left"
def is_right(hand): return hand.classification[0].label == "Right"

# Count fingers for right-hand ON/OFF
def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    states = []

    # Thumb
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
        states.append(1)
    else:
        states.append(0)

    # Others
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            states.append(1)
        else:
            states.append(0)

    return sum(states)



# Start camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_type = results.multi_handedness[i]

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ------------------------------
            # LEFT HAND → BRIGHTNESS (Thumb–Index distance)
            # ------------------------------
            if is_left(hand_type):
                dist, (x1, y1, x2, y2) = thumb_index_distance(hand_landmarks, w, h)

                # NORMALISATION (0 → ~200 px)
                brightness = int(max(0, min(255, dist * 1.3)))  # 1.3 tuning gain

                # Send to Arduino
                arduino.write(f"B{brightness}\n".encode())

                # Show distance line
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"Brightness: {brightness}", (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                print(f"[LEFT] Thumb–Index distance = {dist:.1f} → Brightness = {brightness}")


            # ------------------------------
            # RIGHT HAND → LED ON/OFF
            # ------------------------------
            if is_right(hand_type):
                fingers = count_fingers(hand_landmarks)

                if fingers == 0:
                    arduino.write(b"L0\n")
                    print("[RIGHT] LED OFF")
                else:
                    arduino.write(b"L1\n")
                    print("[RIGHT] LED ON")

    cv2.imshow("Hand Controller", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
