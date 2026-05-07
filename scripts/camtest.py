import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from evaluate import SignModel
import time
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def process_frame(frame):
    global hands
    global mp_hands

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_world_landmarks:
        print(f"No hand landmarks detected, recording 63 zeros.")
        return np.zeros(63)  # Return 63 zeros if no hand landmarks are detected

    for hand_landmarks in results.multi_hand_world_landmarks:
        hand_data = []
        for landmark in hand_landmarks.landmark:
            hand_data.extend([landmark.x, landmark.y, landmark.z])
        return np.array(hand_data)

    raise ValueError(f"Hand landmarks could not be processed for the frame")

def predict_digit(frame, model, scaler):
    features = process_frame(frame).reshape(1, -1)
    features = scaler.transform(features)
    features = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(features)
        probabilities = nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)

    predicted_digit = predicted.item() + 1
    return predicted_digit, probabilities.numpy()

def load_model_and_scaler(language):
    model_path = f'./models/{language}_model.pth'
    scaler_path = f'./models/scaler_{language}.pth'

    model = SignModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    scaler = torch.load(scaler_path)
    return model, scaler

def main():
    input_type = input("Enter the type of input (asl/csl): ").strip().lower()

    if input_type not in ['asl', 'csl']:
        print("Invalid input type. Please enter 'asl' or 'csl'.")
        return

    model, scaler = load_model_and_scaler(input_type)

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

            digit, probabilities = predict_digit(frame, model, scaler)
            print(f"Predicted Digit: {digit}")
            print(f"Probabilities: {probabilities}")

            if input_type == 'asl':
                translation_path = f'./data/show/csl/{digit}.png'
            else:
                translation_path = f'./data/show/asl/{digit}.png'

            translation_image = cv2.imread(translation_path)
            if translation_image is not None:
                translation_image = cv2.resize(translation_image, (200, 200))
                frame[0:200, 0:200] = translation_image

            # Display the frame
            cv2.putText(frame, f"Predicted Digit: {digit}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Hand Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
