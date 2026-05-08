import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def process_frame(frame, landmarker):
    """
    Processes a single frame to extract normalized hand landmarks.
    Normalization: All points are relative to the wrist (Landmark 0).
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Detect hand landmarks
    detection_result = landmarker.detect(mp_image)
    
    # Check if any hand was detected
    if detection_result.hand_world_landmarks:
        # Access the first hand detected
        hand_landmarks = detection_result.hand_world_landmarks[0]
        
        # Wrist (Landmark 0) is the reference point for normalization
        base_x = hand_landmarks[0].x
        base_y = hand_landmarks[0].y
        base_z = hand_landmarks[0].z
        
        hand_data = []
        for lm in hand_landmarks:
            # Centralize the hand by subtracting wrist coordinates
            hand_data.extend([
                lm.x - base_x, 
                lm.y - base_y, 
                lm.z - base_z
            ])
        return hand_data
            
    # Return 63 zeros if no hand is detected
    return [0.0] * 63

def process_images_from_folder(image_folder, landmarker):
    frames = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))]
    features = []

    for frame_path in frames:
        frame = cv2.imread(frame_path)
        if frame is not None:
            hand_data = process_frame(frame, landmarker)
            features.append(hand_data)

    return features

def preprocess_images(output_image_dir, output_features_dir, model_path):
    # Initialize the Hand Landmarker Task
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )
    
    landmarker = vision.HandLandmarker.create_from_options(options)
    
    asl_features_dir = os.path.join(output_features_dir, 'asl_features')
    csl_features_dir = os.path.join(output_features_dir, 'csl_features')

    os.makedirs(asl_features_dir, exist_ok=True)
    os.makedirs(csl_features_dir, exist_ok=True)

    for lang in ['asl', 'csl']:
        for digit in range(1, 12):
            image_folder = os.path.join(output_image_dir, lang, str(digit))
            target_dir = asl_features_dir if lang == 'asl' else csl_features_dir
            features_file = os.path.join(target_dir, f'{digit}.csv')

            features = []

            if os.path.exists(image_folder):
                print(f"Processing: {lang} Digit {digit}...")
                folder_features = process_images_from_folder(image_folder, landmarker)
                features.extend(folder_features)

            if features:
                df = pd.DataFrame(features)
                df.to_csv(features_file, index=False, header=False)
            else:
                print(f"Warning: No valid features found for {image_folder}")

if __name__ == "__main__":
    # Define paths
    data_dir = './data'
    input_img_dir = os.path.join(data_dir, 'sign_images')
    output_feat_dir = os.path.join(data_dir, 'features')
    
    # Ensure this path points to your downloaded .task file
    model_asset_path = './models/hand_landmarker.task' 
    
    if not os.path.exists(model_asset_path):
        print(f"Error: Model file not found at {model_asset_path}")
    else:
        preprocess_images(
            output_image_dir=input_img_dir,
            output_features_dir=output_feat_dir,
            model_path=model_asset_path
        )
        print("Preprocessing completed successfully.")