from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from utils import ChatLLM
from dotenv import load_dotenv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='FakeSV')
args = parser.parse_args()
dataset = args.data
model_name = 'gpt-4o'

src_file = f'data/{dataset}/data.jsonl'
output_dir = f'data/{dataset}/CoT/gpt-4o/'

save_path = os.path.join(output_dir, 'lm_reason.jsonl')

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
Analyze the {text} from a multimodal perspective. Systematically deconstruct the video’s structure and argumentation, identifying any logical flaws or weak points. Focus on elucidating the logical framework without assessing veracity."""
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
        
        lm_text_refine_df = pd.read_json(f'data/{dataset}/CoT/{model_name}/lm_text_refine.jsonl', lines=True, dtype={'vid': str})
        lm_vision_refine_df = pd.read_json(f'data/{dataset}/CoT/{model_name}/lm_visual_refine.jsonl', lines=True, dtype={'vid': str})
        lm_retrieve_df = pd.read_json(f'data/{dataset}/CoT/{model_name}/lm_retrieve.jsonl', lines=True, dtype={'vid': str})
        label_df = pd.read_json(f'data/{dataset}/label.jsonl', lines=True, dtype={'vid': str, 'label': int})

        # select vid in label_df
        src_df = src_df[src_df['vid'].isin(label_df['vid'])]
        src_df = src_df[~src_df['vid'].isin(cur_ids)]
        self.data = src_df
        self.label_df = label_df
        self.lm_text_refine_df = lm_text_refine_df
        self.lm_vision_refine_df = lm_vision_refine_df
        self.lm_retrieve_df = lm_retrieve_df
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        vid = row['vid']
        
        lm_text_refine = self.lm_text_refine_df[self.lm_text_refine_df['vid'] == vid]['text'].values[0]
        lm_vision_refine = self.lm_vision_refine_df[self.lm_vision_refine_df['vid'] == vid]['text'].values[0]
        lm_retrieve = self.lm_retrieve_df[self.lm_retrieve_df['vid'] == vid]['text'].values[0]

        title = row['title'] if 'title' in row else ''
        description = row['description'] if 'description' in row else ''
        title = title + ' ' + description
        
        label = self.label_df[self.label_df['vid'] == vid]['label'].values[0]

        text = f'{lm_text_refine} {lm_vision_refine} {lm_retrieve}'
        return vid, text, label

def customed_collate_fn(batch):
    # preprocess
    # merge to one list
    vids, text, labels = zip(*batch)
    return vids, text, labels


dataloader = DataLoader(MyDataset(), batch_size=4, collate_fn=customed_collate_fn, num_workers=2, shuffle=False)

for batch in tqdm(dataloader):
    vids, texts, labels = batch
    # process inputs
    inputs = [{
        'text': text,
    } for text in texts]
    # process outputs
    outputs = client.chat_batch(inputs)
    # save_dict
    for vid, label, output in zip(vids, labels, outputs):
        save_df = pd.concat([save_df, pd.DataFrame({
            'vid': [vid],
            'text': [output],
            'label': [label]
        })])
    # save to jsonl
    save_df.to_json(save_path, lines=True, orient='records', force_ascii=False)
