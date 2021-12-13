#!/bin/bash
source /home/antpc/anaconda3/etc/profile.d/conda.sh;
CUDA_VISIBLE_DEVICES=1;
conda activate slr
cd ISLR
python lip_extractor.py
conda deactivate