#!/bin/bash
#SBATCH --job-name=lb-training
#SBATCH --export=ALL
#SBATCH --partition=gpu
# Number of GPUs per task 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
# Array Number of Jobs to run in Parallel
# Given via CMD arguments (because it varies depending on the number of jobs)
##SBATCH --array=0-2
#SBATCH --output=logs/%x-%j-%a.log

uv run scripts/train/train.py single stela EN 01 00 lstm --resume