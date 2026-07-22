from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from utils import ChatLLM
from dotenv import load_dotenv
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

save_path = os.path.join(output_dir, 'lm_visual_refine.jsonl')

if os.path.exists(save_path):
    save_df = pd.read_json(save_path, lines=True, dtype={'vid': str})
else:
    save_df = pd.DataFrame(
        columns=['vid', 'text', 'label']
    )

try:
    cur_ids = save_df['vid'].values
except KeyError:
    cur_ids = []

prompt = """
Analyze video frames to generate a descriptive caption, focusing solely on key visual elements and events while ignoring any on-scree-text and subjective elements. 
Based on these images, please infer and describe the content of the video, the main events, and the potential progression of its storyline. Keep the description concise yet comprehensive.
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
                src_file = 'data/FakeSV/data.jsonl'
                src_df = pd.read_json(src_file, lines=True, dtype={'vid': str})
            case 'FakeTT':
                src_file = 'data/FakeTT/data.jsonl'
                src_df = pd.read_json(src_file, lines=True, dtype={'vid': str})
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
        
        image_list = []
        
        for i in range(4):
            image = f'data/{dataset}/quads_4/{vid}_quad_{i}.jpg'
            image = encode_image(image)
            image_list.append(image)

        label = self.label_df[self.label_df['vid'] == vid]['label'].values[0]
        text = ''
        return vid, text, image_list, label

def customed_collate_fn(batch):
    # preprocess
    # merge to one list
    vids, text, images, labels = zip(*batch)
    return vids, text, images, labels


dataloader = DataLoader(MyDataset(), batch_size=1, collate_fn=customed_collate_fn, num_workers=2, shuffle=False)

for batch in tqdm(dataloader):
    vids, texts, images, labels = batch
    # process inputs
    inputs = [{
        'text': text,
        'images': images,
    } for text, images in zip(texts, images)]
    # process outputs
    outputs = client.chat_batch(inputs)
    # save_dict
    for vid, label, output in zip(vids, labels, outputs):
        save_df = pd.concat([save_df, pd.DataFrame({
            'vid': [vid],
            'text': [output],
            'label': [label]
        })], ignore_index=True)
    # save to jsonl
    save_df.to_json(save_path, lines=True, orient='records', force_ascii=False)
