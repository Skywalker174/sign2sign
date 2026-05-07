import os
import cv2
import mediapipe as mp
import pandas as pd
import concurrent.futures

def process_frame(frame, hands):
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_world_landmarks:
        for hand_landmarks in results.multi_hand_world_landmarks:
            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.extend([landmark.x, landmark.y, landmark.z])
            return hand_data
    return [0] * 63  # 21 landmarks * 3 coordinates (x, y, z)

def process_images_from_folder(image_folder, hands):
    frames = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg')]
    features = []

    for frame_path in frames:
        frame = cv2.imread(frame_path)
        hand_data = process_frame(frame, hands)
        features.append(hand_data)

    return features

def preprocess_images(output_image_dir, output_features_dir):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    asl_features_dir = os.path.join(output_features_dir, 'asl_features')
    csl_features_dir = os.path.join(output_features_dir, 'csl_features')

    if not os.path.exists(asl_features_dir):
        os.makedirs(asl_features_dir)
    
    if not os.path.exists(csl_features_dir):
        os.makedirs(csl_features_dir)

    for lang in ['asl', 'csl']:
        for digit in range(1, 12):  # Including digit 11 for exceptions
            image_folder = os.path.join(output_image_dir, lang, str(digit))
            features_file = os.path.join(asl_features_dir if lang == 'asl' else csl_features_dir, f'{digit}.csv')

            features = []

            if os.path.exists(image_folder):
                folder_features = process_images_from_folder(image_folder, hands)
                features.extend(folder_features)

            if features:
                df = pd.DataFrame(features)
                df.to_csv(features_file, index=False)
            else:
                print(f"No valid features found for {image_folder}")

if __name__ == "__main__":
    data_dir = './data'
    preprocess_images(
        output_image_dir=os.path.join(data_dir, 'sign_images'),
        output_features_dir=os.path.join(data_dir, 'features')
    )
