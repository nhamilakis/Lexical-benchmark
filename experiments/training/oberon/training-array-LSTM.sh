#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=%x-%j.log            # Standard output and error log
#SBATCH --array=0-3

# ENV setup, if not set
CODE="/scratch1/projects/lexical-benchmark/v2/jean-zay-code/"


FILENAME="LSTM_train-args.index"


getline_split() {
    if [ $# -ne 2 ]; then
        echo "Usage: getline_split <file> <line_number>"
        return 1
    fi
    
    file="$1"
    n="$2"
    
    if [ ! -f "$file" ]; then
        echo "Error: File '$file' not found"
        return 1
    fi
    
    # Read the line and split by space into global variables
    IFS=' ' read -r TRAIN DEV MODEL <<< $(sed -n "$((n+1))p" "$file")
    
    echo "TRAIN: ${TRAIN}"
    echo "DEV: ${DEV}"
    echo "MODEL: ${MODEL}"
}



getline_split $FILENAME $SLURM_ARRAY_TASK_ID



python $CODE/Lexical_benchmark/src/scripts/train/hf/train_LSTM.py --TrainPath $TRAIN \
    --OutPath $MODEL \
    --ValPath $DEV