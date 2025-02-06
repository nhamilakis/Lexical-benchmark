#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=%x-%j.log             # Standard output and error log
#SBATCH --array=0-3

# ENV setup
DIR_ROOT="/scratch1/projects/lexical-benchmark/v2"
CODE="$DIR_ROOT/jean-zay-code"
DATASET_ROOT="$DIR_ROOT/datasets"
MODEL_ROOT="$DIR_ROOT/models"

FILENAME="${DATASET_ROOT}/script_arg/trans_train-args.index"

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


echo "Running Training of Transformer Model  ($SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID) @ $(date)"

# Grab parameters from index file
read TRAIN DEV MODEL <<< "$(get_line "${JOB_INDEX_FILE}" $SLURM_ARRAY_TASK_ID)"

# Run the Python script
python "$CODE/Lexical_benchmark/src/scripts/train/hf/train_trans.py" \
    --TrainPath "$DATASET_ROOT/$TRAIN" \
    --OutPath "$MODEL_ROOT/$MODEL" \
    --ValPath "$DATASET_ROOT/$DEV" --resume

echo "Completed Training  ($SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID) @ $(date)"