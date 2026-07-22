#!/bin/bash

DATASET=${1:-FakeSV}

# Run textual refining
python preprocess/cot/run_textual_refining.py --data $DATASET

# Run visual refining
python preprocess/cot/run_visual_refining.py --data $DATASET

# Run retrieving
python preprocess/cot/run_retrieving.py --data $DATASET

# Run reasoning
python preprocess/cot/run_reasoning.py --data $DATASET
