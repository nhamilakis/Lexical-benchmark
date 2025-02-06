#!/bin/bash
#SBATCH --job-name=lb-training-LSTM
#SBATCH --account=hhb@a100
# Partition (A100)
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Number of GPUs per task (On a100 8 GPUs per node are available.)
#SBATCH --gres=gpu:4
# Number of cores per task for gpu_p5 (1/8 of 8-GPUs A100 node)
# A100 nodes have 64 cores, should use proportional to GPU number (1 gpu 1/8 of the CPUs)
# For 4 GPUs use 32 cores per task
#SBATCH --cpus-per-task=32
# Only run this when testing
##SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=20:00:00
# Array Number of Jobs to run in Parallel
# Given via CMD arguments (because it varies depending on the number of jobs)
##SBATCH --array=0-2
#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --hint=nomultithread        # hyperthreading is deactivated

export MODEL_ROOT="$WORK/data/models"
export GEN_ROOT="$WORK/data/gen"
export DATASET_ROOT="$WORK/data/datasets"
export CODE="$(pwd)/code"
export JZ=1

if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]; then
    echo "Error: This requires an ARRAY_JOB" >&2
    echo "Add the array option to sbatch: --array=0-2" >&2
    exit 1
fi

if [[ -z "${1}" ]]; then
    echo "Error: index file required" >&2
    exit 1
fi
JOB_INDEX_FILE=$1

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
echo "python: $(uv run which python)"
echo "python-version $(uv run python -V)"
echo "CUDA-AVAILABLE $(uv run python -c 'import torch; print(torch.cuda.is_available());')"

echo "Training LSTM  ($SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID) @ $(date)"

# Grab parameters from index file
read DATASET SPLIT CHUNK VAL_PATH <<< "$(get_line "${JOB_INDEX_FILE}" $SLURM_ARRAY_TASK_ID)"

uv run $CODE/src/scripts/train/hf/train_LSTM.py $DATASET $SPLIT $CHUNK "$DATASET_ROOT/$VAL_PATH"

echo "Completed Training ($SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID) @ $(date)"
