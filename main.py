import cv2
import numpy as np
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7
)

def get_hand_type(hand_landmarks, handedness):
    if handedness.classification[0].label == "Right":
        return "Right"
    else:
        return "Left"

def count_fingers(hand_landmarks, hand_type):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    if hand_type == "Right":
        fingers.append(int(hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x))
    else:
        fingers.append(int(hand_landmarks.landmark[tips_ids[0]].x > hand_landmarks.landmark[tips_ids[0] - 1].x))

    for i in range(1, 5):
        fingers.append(int(hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y))

    return fingers


def draw_animated_hand(image, hand_landmarks, hand_type, finger_count):
    h, w, _ = image.shape
    landmarks = []

    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        landmarks.append((x, y))

    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        cv2.line(image, landmarks[start_idx], landmarks[end_idx], (255, 255, 255), 2)

    for i, point in enumerate(landmarks):
        if i == 12: 
            color = (0, 0, 255) 
        else:
            color = (0, 255, 0) 
        cv2.circle(image, point, 6, color, -1)

    x_center = int(np.mean([lm[0] for lm in landmarks]))
    y_min = int(np.min([lm[1] for lm in landmarks]))

    cv2.putText(image, f"{hand_type}: {sum(finger_count)} fingers",
                (x_center - 70, y_min - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return image

def distance(point1, point2):
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_type = get_hand_type(hand_landmarks, handedness)
            fingers = count_fingers(hand_landmarks, hand_type)

            image = draw_animated_hand(image, hand_landmarks, hand_type, fingers)

            lm = hand_landmarks.landmark
            thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))
            middle_tip = (int(lm[12].x * w), int(lm[12].y * h))

            if fingers == [0, 0, 1, 0, 0]:
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()