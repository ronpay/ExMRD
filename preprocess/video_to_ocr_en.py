import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pytesseract
from Levenshtein import ratio
from autocorrect import Speller
import re
import easyocr
import os
import pandas as pd
from tqdm import tqdm

spell = Speller(lang='en')
reader = easyocr.Reader(['en'], gpu=True)

def extract_frames(video_path, fps=1):
    frames = []
    video = cv2.VideoCapture(video_path)
    video_fps = video.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps / fps)
    
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame)
        frame_count += 1
    
    video.release()
    return frames

def frame_similarity(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

def remove_similar_frames(frames, threshold=0.95):
    if not frames:
        return []
    
    unique_frames = [frames[0]]
    for i in range(1, len(frames)):
        if frame_similarity(frames[i], frames[i-1]) < threshold:
            unique_frames.append(frames[i])
    return unique_frames

def ocr_frames(frames):
    texts = []
    for frame in frames:
        text = reader.readtext(frame, detail = 0)
        text = ' '.join(text)
        # If the text is not empty
        if text:
            texts.append(text.strip())
    return texts

def remove_duplicate_texts(texts, threshold=0.7):
    if not texts:
        return []
    
    unique_texts = [texts[0]]
    for i in range(1, len(texts)):
        if ratio(texts[i], texts[i-1]) < threshold:
            unique_texts.append(texts[i])
    return unique_texts

def clean_and_correct_text(text):
    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
    
    # Spell correction
    words = text.split()
    corrected_words = []
    for word in words:
        if word.isupper():
            corrected_words.append(word)
        else:
            corrected_words.append(spell(word))
    
    # Recombine text
    corrected_text = ' '.join(corrected_words)
    
    return corrected_text.strip()

def extract_text_from_video(video_path):
    # Step 1: Extract one frame per second
    frames = extract_frames(video_path)
    # print(f'Extracted {len(frames)} frames')
    
    # Step 2: Remove similar frames (compared to previous frame)
    unique_frames = remove_similar_frames(frames)
    # print(f'Removed {len(frames) - len(unique_frames)} similar frames')
    
    # Step 3: Perform OCR on each frame
    texts = ocr_frames(unique_frames)
    # Print text
    # for text in texts:
    #     print(f'{text}')
    
    # Step 4: Text cleaning and correction
    cleaned_texts = [clean_and_correct_text(text) for text in texts]
    
    # for text in cleaned_texts:
    #     print(f'{text}')
    
    # Step 5: Remove duplicate text
    unique_texts = remove_duplicate_texts(cleaned_texts)
    
    return unique_texts

# List of datasets to process
datasets = ['FVC']

for dataset in datasets:
    src_dir = f'data/{dataset}/videos'
    dst_file = f'data/{dataset}/ocr.jsonl'

    if not os.path.exists(src_dir):
        print(f"Skipping {dataset}: {src_dir} not found")
        continue

    # Load existing data if available
    if os.path.exists(dst_file):
        dst_df = pd.read_json(dst_file, lines=True)
        cur_ids = dst_df['vid'].values if len(dst_df) > 0 else []
    else:
        dst_df = pd.DataFrame(columns=['vid', 'ocr'])
        cur_ids = []

    bar = tqdm(os.listdir(src_dir))
    for file in bar:
        if not file.endswith('.mp4'):
            continue
            
        video_file = os.path.join(src_dir, file)
        video_id = file.replace('.mp4','')
        bar.set_description(f'Processing {video_id}')
        
        if video_id in cur_ids:
            continue
        
        ocr = ''
        result = extract_text_from_video(video_file)
        for text in result:
            if len(text) > 3:
                ocr += text + '\n'

        tmp_df = pd.DataFrame([{
            'vid': video_id,
            'ocr': ocr}])
        dst_df = pd.concat([dst_df, tmp_df], ignore_index=True)
        dst_df.to_json(dst_file, orient='records', lines=True, force_ascii=False)
