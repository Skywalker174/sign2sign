import os
import cv2

frame_id = 0

def extract_frames(video_path, image_folder, fps=30):
    global frame_id
    frame_count = 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print(f"Error: FPS for video {video_path} is 0")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    skip_interval = max(1, int(video_fps / fps))  # Ensure skip_interval is at least 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if duration > 4 and frame_count % (skip_interval * 2) != 0:  # Skip more frames for videos longer than 4 seconds
            frame_count += 1
            continue

        frame_filename = os.path.join(image_folder, f"frame_{frame_id}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_id += 1
        frame_count += 1

    cap.release()
    print(f"Processed video: {video_path} into {frame_count} frames. Now we have {frame_id} frames in total.")

def process_all_videos(video_dir, output_image_dir):
    global frame_id
    frame_id = 0
    for lang in ['asl', 'csl']:
        lang_folder = os.path.join(video_dir, lang)
        for digit_folder in os.listdir(lang_folder):
            video_folder = os.path.join(lang_folder, digit_folder)
            if not os.path.isdir(video_folder):
                continue

            image_folder = os.path.join(output_image_dir, lang, digit_folder)
            os.makedirs(image_folder, exist_ok=True)

            for video_file in os.listdir(video_folder):
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(video_folder, video_file)
                    extract_frames(video_path, image_folder)

if __name__ == "__main__":
    data_dir = './data'
    process_all_videos(
        video_dir=os.path.join(data_dir, 'sign_videos'),
        output_image_dir=os.path.join(data_dir, 'sign_images')
    )
