from transformers import ViTImageProcessor, ViTModel, CLIPVisionModel, CLIPImageProcessor
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, ChineseCLIPImageProcessor, ChineseCLIPVisionModel, ChineseCLIPTextModel, ChineseCLIPFeatureExtractor
from transformers import BertModel, BertTokenizer,AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import torch
import av
from PIL import Image
from utils import ChatLLM
import loguru
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse
import base64


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='FakeSV')
args = parser.parse_args()
dataset = args.data
model_name = 'gpt-4o'

src_file = f'data/{dataset}/data.jsonl'
output_dir = f'data/{dataset}/CoT/gpt-4o/'
ocr_file = f'data/{dataset}/ocr.jsonl'
transcript_file = f'data/{dataset}/transcript.jsonl'

save_path = os.path.join(output_dir, 'lm_text_refine.jsonl')

if os.path.exists(save_path):
    save_df = pd.read_json(save_path, lines=True, dtype={'vid': str})
else:
    save_df = pd.DataFrame(
        columns=['vid', 'ret', 'label']
    )

try:
    cur_ids = save_df['vid'].values
except KeyError:
    cur_ids = []

prompt = """
Analyze video frames to generate a descriptive caption, focusing solely on key visual elements and events while ignoring any on-scree-text and subjective elements. 
"""
if 'FakeSV' in dataset:
    prompt += "Please answer in Chinese."

load_dotenv(override=True)
client = ChatLLM(
    base_url=os.getenv('OPENAI_BASE_URL'),
    key=os.getenv('OPENAI_API_KEY'),
    prompt=prompt,
    model='gpt-4o-mini',
    temperature=0.7
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class MyDataset(Dataset):
    def __init__(self):
        match dataset:
            case 'FakeSV':
                src_file = 'data/FakeSV/data_complete.jsonl'
                src_df = pd.read_json(src_file, lines=True, dtype={'video_id': str})
                src_df['vid'] = src_df['video_id']
            case 'FakeTT':
                src_file = 'data/FakeTT/data.jsonl'
                src_df = pd.read_json(src_file, lines=True, dtype={'video_id': str})
                src_df['vid'] = src_df['video_id']
            case 'FVC':
                src_file = 'data/FVC/data.jsonl'
                src_df = pd.read_json(src_file, lines=True, dtype={'vid': str})
        
        label_df = pd.read_json(f'data/{dataset}/label.jsonl', lines=True, dtype={'vid': str, 'label': int})

        # select vid in label_df
        src_df = src_df[src_df['vid'].isin(label_df['vid'])]
        src_df = src_df[~src_df['vid'].isin(cur_ids)]
        self.data = src_df
        self.label_df = label_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        vid = row['vid']
        
        image = f'data/{dataset}/quads_4/{vid}.jpg'
        image = encode_image(image)

        label = self.label_df[self.label_df['vid'] == vid]['label'].values[0]
        text = ''
        return vid, text, image, label

def customed_collate_fn(batch):
    # preprocess
    # merge to one list
    vids, text, image, labels = zip(*batch)
    return vids, text, image, labels


dataloader = DataLoader(MyDataset(), batch_size=4, collate_fn=customed_collate_fn, num_workers=2, shuffle=False)

for batch in tqdm(dataloader):
    vids, texts, images, labels = batch
    # process inputs
    inputs = [{
        'text': text,
        'image': image,
    } for text, image in zip(texts, images)]
    # process outputs
    outputs = client.chat_batch(inputs)
    # save_dict
    for vid, label, output in zip(vids, labels, outputs):
        save_df = pd.concat([save_df, pd.DataFrame({
            'vid': [vid],
            'ret': [output],
            'label': [label]
        })])
    # save to jsonl
    save_df.to_json(save_path, lines=True, orient='records', force_ascii=False)
