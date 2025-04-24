#!/bin/bash
#SBATCH --job-name=training
#SBATCH --export=ALL
#SBATCH --partition=gpu
# Number of GPUs per task 
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
# Array Number of Jobs to run in Parallel
# Given via CMD arguments (because it varies depending on the number of jobs)
##SBATCH --array=0-2
#SBATCH --output=logs/%x-%j-%a.log

# Initialize a variable to track if --test was passed
TEST_MODE=false

# Parse all arguments
for arg in "$@"; do
    if [[ "$arg" == "--test" ]]; then
        TEST_MODE=true
    fi
done

if [[ "$TEST_MODE" == true ]]; then
    echo "Running in test mode..."
    uv run code/src/scripts/train/train.py single stela EN 01 00 lstm --resume \
        && echo "training completed succesfully."
    exit 0
fi


# Check if running as part of a job array
if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]; then
    echo "Running as job array task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
    if [[ -z "$1" || ! -f "$1" ]]; then
        echo "Error: Invalid \$1 needs to be an index file"
        exit 1
    fi
    shift
    uv run code/src/scripts/train/train.py array-index $1 "${SLURM_ARRAY_TASK_ID}" $*
else
    echo "Not running as a job array"
    uv run code/src/scripts/train/train.py single $*
fi
echo "training completed succesfully."

