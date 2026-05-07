import os
import cv2
import mediapipe as mp
import time
import sys

# Suppress TensorFlow and other library logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress logs
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Redirect stderr and stdout to suppress warnings and info messages
class NullWriter:
    def write(self, message):
        pass
    def flush(self):
        pass

sys.stderr = NullWriter()
sys.stdout = NullWriter()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

input_video_path = f'./data/videos/input.mp4'
output_video_path = f'./data/videos/output.mp4'

# Check if the input video file exists
if not os.path.isfile(input_video_path):
    sys.stderr = sys.__stderr__
    sys.stdout = sys.__stdout__
    print(f"Error: The video file '{input_video_path}' does not exist.")
    exit()

# Open the input video file
cap = cv2.VideoCapture(input_video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    sys.stderr = sys.__stderr__
    sys.stdout = sys.__stdout__
    print("Error: Could not open video.")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = 560  # Smaller width for faster processing
frame_height = 400  # Smaller height for faster processing
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

# Restore stderr and stdout for normal messages
sys.stderr = sys.__stderr__
sys.stdout = sys.__stdout__
print("Processing...")

# Initialize the MediaPipe Hands model
with mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3) as hands:
    frame_count = 0
    start_time = time.time()  # Start time for processing
    frame_skip = 2  # Process every 3rd frame to speed up

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        frame_count += 1

        # Skip frames
        if frame_count % frame_skip != 0:
            continue

        # Resize the frame
        frame_resized = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Check if hands are detected and draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_resized, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                )

        # Write the processed frame to the output video
        out.write(frame_resized)

    end_time = time.time()  # End time for processing
    processing_time = end_time - start_time
    print(f"Total frames processed: {frame_count}")
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Average FPS: {frame_count / processing_time:.2f}")
    print("Video processing finished successfully.")

# Release video capture and writer objects and close display windows
cap.release()
out.release()