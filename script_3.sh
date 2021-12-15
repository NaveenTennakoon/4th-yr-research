#!/bin/bash
source /home/antpc/anaconda3/etc/profile.d/conda.sh;
CUDA_VISIBLE_DEVICES=1;
conda activate continuous
cd CSLR
tzq config/baseline_10.yml train --continue
conda deactivate