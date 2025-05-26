#!/bin/bash
#SBATCH --job-name=generation
#SBATCH --export=ALL
#SBATCH --partition=gpu
# Number of GPUs per task 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j-%a.log

echo "---START OF GENERATION SCRIPT--- $(date)"

export CLUSTER_NAME="jean-zay"
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
    uv run code/src/scripts/train/generate.py single stela EN 06 00 lstm --temperature-list 0.3,0.6 --hour-per-year "100hpy" \
        && echo "generation completed succesfully."
    exit 0
fi


# Check if running as part of a job array
if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]; then
    echo "Running as job array task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
    if [[ -z "$1" || ! -f "$1" ]]; then
        echo "Error: Invalid \$1 needs to be an index file"
        exit 1
    fi
    INDEX_FILE="$1"
    shift
    uv run code/src/scripts/train/generate.py array-index "${INDEX_FILE}" "${SLURM_ARRAY_TASK_ID}" $* \
        && echo "generation completed succesfully."
else
    echo "Not running as a job array"
    echo ">train.py single $*"
    uv run code/src/scripts/train/generate.py single $* \
        && echo "generation completed succesfully."
fi
echo "---END OF GENERATION SCRIPT--- $(date)"