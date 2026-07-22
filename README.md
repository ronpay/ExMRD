# <center> Following Clues, Approaching the Truth: Explainable Micro-Video Rumor Detection via Chain-of-Thought Reasoning </center>

<div align="center">
  <img src="https://img.shields.io/badge/WWW-2025-blue" alt="Static Badge">
</div>

This repo provides the official implementation of ExMRD as described in the paper:

*Following Clues, Approaching the Truth: Explainable Micro-Video Rumor Detection via Chain-of-Thought Reasoning* **(WWW'25 research track)**


# Source Code Structure

```bash
data        # dir of each dataset
- FakeSV
- FakeTT
- FVC

preprocess  # code to preprocess video and CoT preprocess

src
- config    # training config
- model     # ExMRD model
- utils     # training utils
- data      # dataloader of ExMRD
```

# Dataset

We provide video IDs for each dataset in both temporal and five-fold splits. Due to copyright restrictions, the raw datasets are not included. You can obtain the datasets from their respective original project sites.

### FakeSV

Access the full dataset from [ICTMCG/FakeSV: Official repository for "FakeSV: A Multimodal Benchmark with Rich Social Context for Fake News Detection on Short Video Platforms", AAAI 2023.](https://github.com/ICTMCG/FakeSV).

### FakeTT

Access the full dataset from [ICTMCG/FakingRecipe: Official Repository for "FakingRecipe: Detecting Fake News on Short Video Platforms from the Perspective of Creative Process", ACM MM 2024](https://github.com/ICTMCG/FakingRecipe).

### FVC

Access the full dataset from [MKLab-ITI/fake-video-corpus: A dataset of debunked and verified user-generated videos.](https://github.com/MKLab-ITI/fake-video-corpus).

# Start

## Environment Setup

To set up the environment, run the following commands:

```bash
# install ffmpeg (if you are using a Debian-based OS, run the following command)
apt install ffmpeg 
# create env using conda
conda create --name ExMRD python=3.12
conda activate ExMRD
pip install -r requirements.txt
```

## Prepare Datasets

1. Get raw dataset (including videos and metadatas) from source, and save raw videos to `{dataset}/videos`.
2. Init `{dataset}/data.jsonl` for each dataset, with each line containing a `vid`(video id) to include all video ids in full dataset.
3. Init `{dataset}/label.jsonl` for each dataset with each line containing a `vid` and its associated `label` (1 or 0).

## Video Data Preprocessing

Run the following command:
```bash 
bash run/preprocess.sh
```

Or you can manually preprocess data following these instructions:

1. Sample 16 frames from each video in the dataset and store them in `{dataset}/frames_16`.

2. Extract on-screen text from keyframes using Paddle-OCR and save to `{dataset}/ocr.jsonl`.

3. Extract audio transcripts from video audio using Whisper and save to `{dataset}/transcript.jsonl`.

4. Extract visual features from each video using a pre-trained CLIP-ViT model and save to `{dataset}/fea/vit_tensor.pt`.

## R<sup>3</sup>CoT with MLLM Preprocessing

Run the following commands:
```bash
# setup .env.example, and make sure you have access to OpenAI API
mv .env.example .env
# defaults to FakeSV; pass FakeTT or FVC to process another dataset
bash run/cot.sh FakeSV
```

## Run

```bash
# Run ExMRD for the FakeSV dataset
python src/main.py --config-name ExMRD_FakeSV

# Run ExMRD for the FakeTT dataset
python src/main.py --config-name ExMRD_FakeTT

# Run ExMRD for the FVC dataset
python src/main.py --config-name ExMRD_FVC
```

# Citation

```bib
@inproceedings{hong2025following,
	author = {Hong, Rongpei and Lang, Jian and Xu, Jin and Cheng, Zhangtao and Zhong, Ting and Zhou, Fan},
	booktitle = {The {Web} {Conference} ({WWW})},
	year = {2025},
	organization = {ACM},
	title = {Following Clues, Approaching the Truth: Explainable Micro-Video Rumor Detection via Chain-of-Thought Reasoning},
}
```