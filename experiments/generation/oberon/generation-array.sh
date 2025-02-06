#!/bin/bash
#SBATCH --job-name=lb-generation
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

export PROJECT_DIR="/scratch1/projects/lexical-benchmark/v2"
export MODEL_ROOT="$PROJECT_DIR/models"
export GEN_ROOT="$PROJECT_DIR/gen/oberon"
export DATASET_ROOT="$PROJECT_DIR/datasets"
export CODE="$HOME/workspace/src/LexicalBenchmark2/source"

if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]; then
    echo "Error: This requires an ARRAY_JOB" >&2
    echo "Add the array option to sbatch: --array=0-2" >&2
    exit 1
fi
JOB_INDEX_FILE=$1


if [[ -z "${1}" ]]; then
    echo "Error: index file required" >&2
    exit 1
fi

get_line() {
    local file="$1"
    local n="$2"

    # Check if both arguments are provided
    if [[ $# -ne 2 ]]; then
        echo "Usage: get_line <file> <line_number>" >&2
        return 1
    fi

    # Check if file exists
    if [[ ! -f "$file" ]]; then
        echo "Error: File '$file' not found" >&2
        return 1
    fi

    # Check if n is a number
    if ! [[ "$n" =~ ^[0-9]+$ ]]; then
        echo "Error: Line number must be a non-negative integer" >&2
        return 1
    fi

    # Get the line (adding 1 because sed uses 1-based indexing)
    n=$((n + 1))
    sed "${n}q;d" "$file"
}

echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "CPUs per Node: $SLURM_CPUS_ON_NODE"
echo "Allocated GPUs: $SLURM_GPUS"
echo "GPU List: $CUDA_VISIBLE_DEVICES"

# Print system information
echo -e "\n=== System Information ==="
echo "Hostname: $(hostname)"
echo "CPU Info: $(lscpu | grep 'Model name' | sed 's/Model name: *//')"
echo "Memory Info: $(free -h | grep Mem)"

echo -e "\n=== PYTHON ==="
echo "python: $(which python)"
echo "python-version $(python -V)"
echo "CUDA-AVAILABLE $(python -c 'import torch; print(torch.cuda.is_available());')"


echo "Running Generation  ($SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID) @ $(date)"

# Grab parameters from index file
read model output <<< "$(get_line "${JOB_INDEX_FILE}" $SLURM_ARRAY_TASK_ID)"

python $CODE/src/scripts/generation/generate.py --gen_file "$GEN_ROOT/CHILDES_model.csv" --model_path "$MODEL_ROOT/$model" --generation_path "$GEN_ROOT/$output" --save_interval 100 --resume

echo "Completed Generation  ($SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID) @ $(date)"