#!/usr/bin/env python
"""
Gesture-based Volume Control using MediaPipe and OpenCV
Controls system volume based on thumb-index finger distance
"""

import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Audio control setup
devices = AudioUtilities.GetSpeakers()
volume_ctrl = devices.EndpointVolume
vol_range = volume_ctrl.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

# Start webcam
cap = cv2.VideoCapture(0)

# Mute/unmute gesture settings
mute_state = False
mute_hold_counter = 0
unmute_hold_counter = 0
HOLD_FRAMES = 15
MUTE_DIST_THRESHOLD = 30
UNMUTE_DIST_THRESHOLD = 150

print("Gesture Volume Control started!")
print("Press 'q' to quit")
print("Use thumb and index finger pinch distance to control volume")

while True:
    success, img = cap.read()
    if not success:
        break
    
    h, w, c = img.shape
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))
            
            if len(lm_list) >= 9:
                x1, y1 = lm_list[4]   # Thumb tip
                x2, y2 = lm_list[8]   # Index tip
                distance = np.hypot(x2 - x1, y2 - y1)

                # Mute / unmute by holding pinch (thumb-index) or holding open
                if distance < MUTE_DIST_THRESHOLD:
                    mute_hold_counter += 1
                    unmute_hold_counter = 0
                    if mute_hold_counter >= HOLD_FRAMES and not mute_state:
                        mute_state = True
                        volume_ctrl.SetMute(1, None)
                elif distance > UNMUTE_DIST_THRESHOLD:
                    unmute_hold_counter += 1
                    mute_hold_counter = 0
                    if unmute_hold_counter >= HOLD_FRAMES and mute_state:
                        mute_state = False
                        volume_ctrl.SetMute(0, None)
                else:
                    mute_hold_counter = 0
                    unmute_hold_counter = 0

                # Map distance to volume (keeps underlying level)
                vol = np.interp(distance, [30, 200], [min_vol, max_vol])
                volume_ctrl.SetMasterVolumeLevel(vol, None)

                # Draw feedback
                cv2.circle(img, (x1, y1), 10, (255, 0, 0), -1)
                cv2.circle(img, (x2, y2), 10, (255, 0, 0), -1)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, f"Distance: {int(distance)}", (50, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Show mute status
                if mute_state:
                    cv2.putText(img, "MUTED", (50, 95), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)
    else:
        # reset hold counters when no hand is visible
        mute_hold_counter = 0
        unmute_hold_counter = 0
    cv2.imshow("Gesture Volume Control", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
