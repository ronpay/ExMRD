import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def process_frames(input_folder, output_folder):
    if not os.path.exists(input_folder):
        print(f"Skipping: input folder does not exist: {input_folder}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed_count = 0
    error_count = 0

    video_folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
    
    for vid_folder in tqdm(video_folders, desc="Processing video folders", unit="folders"):
        frames_folder = os.path.join(input_folder, vid_folder)
        
        frames = []
        for i in range(16):
            frame_path = os.path.join(frames_folder, f"frame_{i:03d}.jpg")
            if os.path.exists(frame_path):
                frame = Image.open(frame_path)
                frames.append(frame)
            else:
                tqdm.write(f"Cannot find frame: {frame_path}")
        
        if len(frames) < 16:
            tqdm.write(f"Folder {vid_folder} contains fewer than 16 frames")
            error_count += 1
            continue

        # Create 4 images with 2x2 grid layout
        for quad_index in range(4):
            start_index = quad_index * 4
            quad_frames = frames[start_index:start_index+4]
            
            min_width = min(frame.width for frame in quad_frames)
            min_height = min(frame.height for frame in quad_frames)
            quad_frames = [frame.resize((min_width, min_height)) for frame in quad_frames]

            grid = Image.new('RGB', (min_width * 2, min_height * 2))
            grid.paste(quad_frames[0], (0, 0))
            grid.paste(quad_frames[1], (min_width, 0))
            grid.paste(quad_frames[2], (0, min_height))
            grid.paste(quad_frames[3], (min_width, min_height))

            output_path = os.path.join(output_folder, f"{vid_folder}_quad_{quad_index}.jpg")
            grid.save(output_path, 'JPEG')

        processed_count += 1

    print(f"Number of successfully processed video folders: {processed_count}")
    print(f"Number of failed video folders: {error_count}")

# Process multiple datasets (FakeSV, FakeTT, FVC)
datasets = ["FakeSV", "FakeTT", "FVC"]

for dataset in datasets:
    print(f"Processing {dataset} dataset...")
    input_folder = f'data/{dataset}/frames_16'
    output_folder = f'data/{dataset}/quads_4'
    process_frames(input_folder, output_folder)