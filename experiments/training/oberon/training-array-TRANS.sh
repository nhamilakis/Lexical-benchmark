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
    sed -n "${n}p" "$file"
}

# Ensure SLURM_ARRAY_TASK_ID is set
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Make sure this script is run with an SLURM array." >&2
    exit 1
fi

# Retrieve parameters from the index file
LINE=$(get_line "$FILENAME" "$SLURM_ARRAY_TASK_ID")
if [[ $? -ne 0 || -z "$LINE" ]]; then
    echo "Error: Unable to retrieve line from file '$FILENAME' for task ID $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

# Debugging: Print the retrieved line
echo "Retrieved line: $LINE"

# Split the line into variables
IFS=' ' read -r TRAIN DEV MODEL <<< "$LINE"

# Debugging: Print the parsed variables
echo "Parsed variables:"
echo "TRAIN=$TRAIN"
echo "DEV=$DEV"
echo "MODEL=$MODEL"

# Run the Python script
python "$CODE/Lexical_benchmark/src/scripts/train/hf/train_trans.py" \
    --TrainPath "$DATASET_ROOT/$TRAIN" \
    --OutPath "$MODEL_ROOT/$MODEL" \
    --ValPath "$DATASET_ROOT/$DEV"
