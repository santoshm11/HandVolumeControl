import cv2
import mediapipe as mp
import math
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# Setup system audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))

# Setup hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

MIN_DISTANCE = 20
MAX_DISTANCE = 200

def map_distance_to_volume(distance, min_d, max_d):
    distance = max(min_d, min(max_d, distance))
    volume = np.interp(distance, [min_d, max_d], [0.0, 1.0])  # volume range: 0.0 - 1.0
    return volume

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape

        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))

        cv2.circle(frame, thumb_pos, 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(frame, index_pos, 10, (255, 0, 0), cv2.FILLED)
        cv2.line(frame, thumb_pos, index_pos, (255, 0, 0), 3)

        distance = calculate_distance(thumb_pos, index_pos)
        volume = map_distance_to_volume(distance, MIN_DISTANCE, MAX_DISTANCE)

        # Set system volume
        volume_ctrl.SetMasterVolumeLevelScalar(volume, None)

        volume_percent = int(volume * 100)
        cv2.putText(frame, f'Volume: {volume_percent}%', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
