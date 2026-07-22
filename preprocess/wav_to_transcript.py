import pandas as pd
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

batch_size = 4

# Initialize ASR model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    torch_dtype=torch_dtype,
    batch_size=batch_size,
    device=device,
)

# Dataset class remains unchanged
class AudioDataset(Dataset):
    def __init__(self, df, audio_dir):
        self.df = df
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row['vid']
        audio_file = os.path.join(self.audio_dir, f"{video_id}.wav")
        return {"vid": video_id, "audio_file": audio_file}

# Modified process_batch function
def process_batch(batch, pipe):
    video_ids = batch['vid']
    audio_files = batch['audio_file']
    results = []
    valid_audio_files = []
    valid_video_ids = []

    for vid, af in zip(video_ids, audio_files):
        if os.path.exists(af):
            valid_audio_files.append(af)
            valid_video_ids.append(vid)
        else:
            results.append((vid, ""))

    if valid_audio_files:
        transcripts = pipe(valid_audio_files, batch_size=len(valid_audio_files))
        transcripts = [r['text'] for r in transcripts]
        results.extend(zip(valid_video_ids, transcripts))
    
    return sorted(results, key=lambda x: video_ids.index(x[0]))

# Main processing function
def process_dataset(dataset_name):
    src_file = f'data/{dataset_name}/data.jsonl'
    if not os.path.exists(src_file):
        print(f"Skipping {dataset_name}: {src_file} not found")
        return
    src_df = pd.read_json(src_file, lines=True, dtype={'vid': str})

    dst_file = f'data/{dataset_name}/transcript.jsonl'

    if not os.path.exists(dst_file):
        dst_df = pd.DataFrame(columns=['vid', 'transcript'])
        dst_df.to_json(dst_file, orient='records', lines=True)
    else:
        dst_df = pd.read_json(dst_file, lines=True, dtype={'vid': str})
        
    cur_ids = set(dst_df['vid'].values) if len(dst_df) > 0 else set()

    # Filter already processed data
    src_df = src_df[~src_df['vid'].isin(cur_ids)]

    # Create Dataset and DataLoader
    dataset = AudioDataset(src_df, f'data/{dataset_name}/audios')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Process data
    for batch in tqdm(dataloader, total=len(dataloader), desc=f"Processing {dataset_name}"):
        results = process_batch(batch, pipe)
        for video_id, transcript in results:
            tmp_df = pd.DataFrame([{'vid': video_id, 'transcript': transcript}])
            dst_df = pd.concat([dst_df, tmp_df], ignore_index=True)
        
        dst_df.to_json(dst_file, orient='records', lines=True, force_ascii=False)

    print(f"Processing {dataset_name} complete!")

# Process each dataset
datasets = ["FakeSV", "FakeTT", "FVC"]
for dataset_name in datasets:
    print(f"Starting to process {dataset_name}...")
    process_dataset(dataset_name)

print("All datasets processed!")