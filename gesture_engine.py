import cv2
import mediapipe as mp
import numpy as np
import time
from pycaw.pycaw import AudioUtilities
import matplotlib.pyplot as plt
import io
import base64


class GestureEngine:

    def __init__(self):

        # Detection parameters
        self.detect_conf = 0.7
        self.track_conf = 0.5
        self.max_hands = 1

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.create_hands()
        self.mp_draw = mp.solutions.drawing_utils

        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Audio Control
        devices = AudioUtilities.GetSpeakers()
        self.volume_ctrl = devices.EndpointVolume
        self.min_vol, self.max_vol = self.volume_ctrl.GetVolumeRange()[:2]

        # Performance tracking
        self.prev_time = 0
        self.fps = 0

        # Volume tracking
        self.current_volume = 0
        self.volume_history = []
        self.distance_history = []
        self.frame_count = 0

    # 🔹 Create MediaPipe Hands
    def create_hands(self):
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detect_conf,
            min_tracking_confidence=self.track_conf
        )

    # 🔹 Update detection parameters dynamically
    def update_params(self, detect, track, max_hands):
        self.detect_conf = float(detect)
        self.track_conf = float(track)
        self.max_hands = int(max_hands)
        self.create_hands()

    # 🔹 Distance → Volume Mapping (0–100%)
    def map_distance_to_volume(self, distance, min_dist=30, max_dist=200):
        distance = np.clip(distance, min_dist, max_dist)
        normalized = (distance - min_dist) / (max_dist - min_dist)
        return int(normalized * 100)

    # 🔹 Main Frame Processing
    def get_frame(self):

        success, img = self.cap.read()
        if not success:
            return None, {}

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Processing time for latency
        start = time.time()
        results = self.hands.process(rgb)
        latency = int((time.time() - start) * 1000)

        hands_count = 0
        gesture = "None"
        distance = 0

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:

                hands_count += 1

                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    img, hand, self.mp_hands.HAND_CONNECTIONS
                )

                lm = hand.landmark
                x1, y1 = int(lm[4].x * w), int(lm[4].y * h)  # Thumb tip
                x2, y2 = int(lm[8].x * w), int(lm[8].y * h)  # Index tip

                distance = np.hypot(x2 - x1, y2 - y1)

                # 🔹 Gesture Classification
                if distance < 40:
                    gesture = "Pinch"
                elif distance < 120:
                    gesture = "Neutral"
                else:
                    gesture = "Spread"

                # 🔹 Volume Mapping
                volume_percent = self.map_distance_to_volume(distance)

                vol_db = np.interp(
                    volume_percent,
                    [0, 100],
                    [self.min_vol, self.max_vol]
                )

                self.volume_ctrl.SetMasterVolumeLevel(vol_db, None)
                self.current_volume = volume_percent

                # 🔹 Store History for Graphs
                self.frame_count += 1
                self.volume_history.append(self.current_volume)
                self.distance_history.append(distance)

                if len(self.volume_history) > 100:
                    self.volume_history.pop(0)
                    self.distance_history.pop(0)

        # 🔹 FPS Calculation
        cur = time.time()
        self.fps = int(1 / (cur - self.prev_time)) if self.prev_time else 0
        self.prev_time = cur

        stats = {
            "hands": hands_count,
            "gesture": gesture,
            "distance": int(distance),
            "volume": self.current_volume,
            "fps": self.fps,
            "latency": f"{latency} ms",
            "resolution": f"{w}x{h}"
        }

        return img, stats

    # 🔹 Static Mapping Graph
    def generate_mapping_graph(self):

        distances = np.linspace(30, 200, 50)
        volumes = [self.map_distance_to_volume(d) for d in distances]

        plt.figure(figsize=(5, 3))
        plt.plot(distances, volumes)
        plt.xlabel("Gesture Distance (px)")
        plt.ylabel("Volume (%)")
        plt.title("Distance → Volume Mapping")
        plt.grid(True)

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return image_base64

    # 🔹 Real-Time Variation Graph
    def generate_variation_graph(self):

        plt.figure(figsize=(6, 4))

        x = range(len(self.volume_history))

        # Volume variation
        plt.subplot(2, 1, 1)
        plt.plot(x, self.volume_history, color='green')
        plt.title("Real-Time Volume Variation")
        plt.ylabel("Volume (%)")
        plt.grid(True)

        # Distance variation
        plt.subplot(2, 1, 2)
        plt.plot(x, self.distance_history, color='blue')
        plt.title("Real-Time Distance Variation")
        plt.ylabel("Distance (px)")
        plt.xlabel("Frame Index")
        plt.grid(True)

        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return image_base64