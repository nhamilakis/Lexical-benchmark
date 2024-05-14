#!/bin/bash
#SBATCH --job-name=generate_100
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --output=100_0.8.out # output file name
#SBATCH --error=100_0.8.err  # error file name
#SBATCH --time=96:00:00

module load miniconda3/latest
conda init
conda activate babylm_train


python generation.py --ModelPath /home/jliu/STELAWord/models/char_with/100h/00 \
    --DictPath /home/jliu/STELAWord/data/preprocessed/EN/100h/00/bin_with \
    --DataPath /scratch2/jliu/freq_bias_benchmark/data/generation/unprompted/sample_random/generated/matched/100.csv \
    --temp_lst 0.8 \
    --OutputPath /scratch2/jliu/freq_bias_benchmark/data/generation/unprompted/sample_random/generated/matched1 \
    --gpu True