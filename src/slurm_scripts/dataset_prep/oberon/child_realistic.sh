#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=childrealistic-prep
#SBATCH --time=0:45:00
#SBATCH --export=ALL
#SBATCH --output %x-%J.log

CODE="${CODE:-source}"
uv run $CODE/src/scripts/dataset_clean/child_realistic.py $@