#!/bin/bash
#SBATCH --job-name=gen_array
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=%x-%j.log            # Standard output and error log
#SBATCH --array=0-5



FILENAME="STELA.gen"

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
    fi  # Changed this closing brace from } to fi
    
    # Read the line and split by comma into global variables
    IFS=',' read -r MODEL_ROOT GEN_ROOT <<< $(sed -n "$((n+1))p" "$file")

    echo "MODEL_ROOT: ${MODEL_ROOT}"
    echo "GEN_ROOT: ${GEN_ROOT}"
}



getline_split $FILENAME $SLURM_ARRAY_TASK_ID



python generate.py --model_path $MODEL_ROOT \
    --generation_path $GEN_ROOT \
    --debug "False"
