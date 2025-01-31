import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Inisialisasi mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi pycaw untuk kontrol volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Dapatkan range volume
volume_range = volume.GetVolumeRange()
min_vol = volume_range[0]
max_vol = volume_range[1]

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame secara horizontal untuk mirror effect
    frame = cv2.flip(frame, 1)

    # Konversi frame ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Proses frame dengan mediapipe hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark tangan
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Dapatkan koordinat landmark jari telunjuk dan jempol
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Hitung jarak antara ujung jari telunjuk dan jempol
            distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

            # Map jarak ke range volume
            vol = np.interp(distance, [0.05, 0.3], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)

            # Tampilkan volume pada frame
            cv2.putText(frame, f"Volume: {int(np.interp(vol, [min_vol, max_vol], [0, 100]))}%", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Tampilkan frame
    cv2.imshow('Hand Tracking Volume Control', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()