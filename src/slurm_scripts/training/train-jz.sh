#!/bin/bash
#SBATCH --job-name=training
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

export JZ=1


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