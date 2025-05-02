import numpy as np
import pandas as pd
import cv2
import moviepy.editor as mp
import os

metadata = "HealthTerms.xlsx"
input_folder = "HealthVideos"
output_folder = "AugHealthVideos"
resized_output_folder = "ResizedVideos_224"
os.makedirs(resized_output_folder, exist_ok=True)

df = pd.read_excel(metadata)
os.makedirs(output_folder, exist_ok=True)

def color_shift_clip(video, shift_value=10):
    def process(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift_value) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return video.fl_image(process)


def change_speed(video, factor):
    return video.fx(mp.vfx.speedx, factor)

def adjust_brightness(video, factor):
    return video.fl_image(lambda frame: np.clip(frame * factor, 0, 255).astype(np.uint8))

def rotate_video(video, angle):
    return video.rotate(angle)

def crop_and_resize(video, crop_factor=0.9):
    w, h = video.size
    return video.crop(x1=w*(1-crop_factor)//2, y1=h*(1-crop_factor)//2, x2=w*(1+crop_factor)//2, 
                      y2=h*(1+crop_factor)//2).resize((w, h))

def adjust_contrast(video, factor):
    return video.fx(mp.vfx.lum_contrast, contrast=factor)

def tint_color(video, multipliers):
    def process(frame):
        r, g, b = multipliers
        frame = frame.astype(np.float32)
        frame[..., 0] *= b  
        frame[..., 1] *= g  
        frame[..., 2] *= r  
        return np.clip(frame, 0, 255).astype(np.uint8)
    
    return video.fl_image(process)

def increase_saturation_clip(video, scale=1.5):
    def process(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= scale
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return video.fl_image(process)

def process_augmentation():
    for _, row in df.iterrows():
        term = row["Health_Term"]
        video_file = row["Video_File"]
        
        input_path = os.path.join(input_folder, video_file)
        term_folder = os.path.join(output_folder, term.replace(".mp4", ""))
        os.makedirs(term_folder, exist_ok=True)
        
        # Load video
        try:
            video = mp.VideoFileClip(input_path)
        except Exception as e:
            print(f"Failed to load {video_file}: {e}")
            continue
        
        # Save original
        video.write_videofile(os.path.join(term_folder, video_file), codec="libx264", audio=False)
        
        augmentations = [
            (change_speed, (1.2,)),
            (adjust_brightness, (1.2,)),
            (adjust_brightness, (0.8,)),
            (rotate_video, (2,)),
            (rotate_video, (-2,)),
            (crop_and_resize, (0.9,)),
            (adjust_contrast, (1.2,)),
            (color_shift_clip, (15,)),
            (tint_color, ((1.2, 1.0, 1.0),)),
            (increase_saturation_clip, (1.5,))
        ]

        
        for i, (func, args) in enumerate(augmentations):
            aug_video = func(video, *args)
            output_path = os.path.join(term_folder, f"aug_{i+1}_" + video_file)
            aug_video.write_videofile(output_path, codec="libx264", audio=False)
        
        print(f"Processed {term}: {video_file}")

# FRAME PROCESSING 
def load_video(input_folder):
    cap = cv2.VideoCapture(input_folder)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames)

def resize_frames(video, target_size):
    return [cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4) for frame in video]


def save_video_to_folder(video_frames, output_path, term_name, video_id, fps=25):
    term_folder = os.path.join(output_path, term_name)
    os.makedirs(term_folder, exist_ok=True)
    save_path = os.path.join(term_folder, f"{video_id}.mp4")

    height, width, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))  
    
    for frame in video_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()

# MAIN PROCESSING
def resize_and_save(input_folder, output_folder, target_size):
    total_videos = 0
    for term_name in os.listdir(input_folder):
        term_path = os.path.join(input_folder, term_name)
        if not os.path.isdir(term_path):
            continue

        for video_file in os.listdir(term_path):
            if not video_file.endswith('.mp4'):
                continue

            video_path = os.path.join(term_path, video_file)
            frames = load_video(video_path)

            if len(frames) == 0:
                print(f"[ERROR] Couldn't load frames from: {video_path}")
                continue

            resized = resize_frames(frames, target_size)

            video_id = os.path.splitext(video_file)[0]
            save_video_to_folder(resized, output_folder, term_name, video_id)
            total_videos += 1

    print(f"\nDone processing {total_videos} videos into '{output_folder}'.")

# CONFIG 
target_size = (224, 224)

# Run Augmentation
#process_augmentation()

# Run Resizing and Padding
resize_and_save(output_folder, resized_output_folder, target_size)
