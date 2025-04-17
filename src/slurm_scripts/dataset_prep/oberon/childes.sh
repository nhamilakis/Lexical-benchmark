#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=childes-prep
#SBATCH --time=0:45:00
#SBATCH --export=ALL
#SBATCH --output %x-%J.log

CODE="${CODE:-source}"
HARDCODED_ARGS=(
    "--save_args" "--skip_word_cleaning" "--skip_word_frequencies" 
    "/lustre/fswork/projects/rech/hhb/ucx81cx/data2/datasets/CHILDES"
)



uv run $CODE/src/scripts/dataset_clean/childes.py "${HARDCODED_ARGS[@]}" "$@"