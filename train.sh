#!/bin/bash
#
#SBATCH --job-name=train_rel
#SBATCH --output=train_rel%j.out
#SBATCH --error=train_rel%j.err
#
#SBATCH --partition=gpu_batch   # Using the GPU partition
#SBATCH --gres=gpu:1            # Requesting 1 GPUs
#SBATCH --cpus-per-gpu=8        # Allocating 8 CPUs per GPU
#SBATCH --mem-per-gpu=80G       # Allocating 80 GB memory per GPU
#SBATCH --time=1-00:00:00       # Max run time of 4 days
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user="fang.wu@arcinstitute.org"

# Activate your conda environment if needed
source /opt/conda/etc/profile.d/conda.sh
conda activate llm   # Replace with your actual conda environment

# Run the Python script
python main.py  --dataset=rel-amazon --task=user-ltv --epochs=10 --batch_size=2 --val_size=2 --lr=0.0001 --wd=0.0015 --dropout=0.2 --val_steps=1000 --context --context_table=review
