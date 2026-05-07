import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from evaluate import SignModel
import time
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def process_image(image_path):
    global hands
    global mp_hands
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image {image_path} not found")

    # Process image to extract hand landmarks using Mediapipe

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_world_landmarks:
        print(f"No hand landmarks detected in image {image_path}, recording 63 zeros.")
        return np.zeros(63)  # Return 63 zeros if no hand landmarks are detected

    for hand_landmarks in results.multi_hand_world_landmarks:
        hand_data = []
        for landmark in hand_landmarks.landmark:
            hand_data.extend([landmark.x, landmark.y, landmark.z])
        return np.array(hand_data)

    raise ValueError(f"Hand landmarks could not be processed for image {image_path}")

def predict_digit(image_path, model_path, scaler_path):
    print("Process image time")
    start = time.time()
    # Process the image to extract features
    features = process_image(image_path).reshape(1, -1)
    end = time.time()
    print(end - start)

    print("Load model time")
    start = time.time()
    # Load the trained model and scaler
    model = SignModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    end = time.time()
    print(end - start)

    print("Load scaler time")
    start = time.time()
    scaler = torch.load(scaler_path)
    features = scaler.transform(features)
    end = time.time()
    print(end - start)

    print("Predict time")
    start = time.time()
    features = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(features)
        probabilities = nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)
    end = time.time()
    print(end - start)


    predicted_digit = predicted.item() + 1
    return predicted_digit, probabilities.numpy()

if __name__ == "__main__":
    image_path = './data/extest/hand_gesture.png'  # Path to the hand gesture image
    model_path = './models/asl_model.pth'
    scaler_path = './models/scaler_asl.pth'

    try:
        digit, probabilities = predict_digit(image_path, model_path, scaler_path)
        print(f"Predicted Digit: {digit}")
        print(f"Probabilities: {probabilities}")
    except Exception as e:
        print(f"Error: {e}")
