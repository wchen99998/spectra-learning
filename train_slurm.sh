#!/bin/bash

#SBATCH --job-name=process_data
#SBATCH --gres=gpu:1            # Request all GPUs on the node (adjust number)
#SBATCH --time=12:00:00         # Set a 24-hour time limit
cd ~/Project/spectra-learning
source .venv/bin/activate


python train.py --config configs/gems_small.py --workdir experiments/gems_small_bsp_sigreg_norm_2