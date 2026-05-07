import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from evaluate import SignModel
import time
import mediapipe as mp
import os
import socket
import json

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def process_frame(frame):
    global hands
    global mp_hands

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_world_landmarks:
        return np.zeros((21, 3))  # Return 21 points with 3 coordinates (x, y, z)

    for hand_landmarks in results.multi_hand_world_landmarks:
        hand_data = []
        for landmark in hand_landmarks.landmark:
            hand_data.append([landmark.x, landmark.y, landmark.z])
        return np.array(hand_data)

    return np.zeros((21, 3))

def main():
    # Set up UDP connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('localhost', 5005)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            hand_points = process_frame(frame).tolist()
            message = json.dumps(hand_points)
            sock.sendto(message.encode(), server_address)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
