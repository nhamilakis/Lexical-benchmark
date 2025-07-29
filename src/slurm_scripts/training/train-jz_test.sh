#!/bin/bash
#SBATCH --job-name=training
#SBATCH --account=hhb@h100
# Partition (H100)
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Number of GPUs per task (On a100 8 GPUs per node are available.)
#SBATCH --gres=gpu:2
# Number of cores per task for gpu_p5 (1/8 of 8-GPUs A100 node)
# A100 nodes have 64 cores, should use proportional to GPU number (1 gpu 1/8 of the CPUs)
# For 4 GPUs use 32 cores per task
#SBATCH --cpus-per-task=16
# Only run this when testing
#SBATCH --time=20:00:00
# Array Number of Jobs to run in Parallel
# Given via CMD arguments (because it varies depending on the number of jobs)
#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --hint=nomultithread        # hyperthreading is deactivated
echo "---START OF TRAIN SCRIPT--- $(date)"
# All files created belong to the project
umask 007

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
    uv run code/src/scripts/train/train.py single stela EN 01 00 lstm --batch-size 256 \
        && echo "training completed succesfully."
    exit 0
fi


# Check if running as part of a job array
if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]; then
    echo ">train.py array-index "${INDEX_FILE}" "${SLURM_ARRAY_TASK_ID}" $*"
    echo "Running as job array task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
    if [[ -z "$1" || ! -f "$1" ]]; then
        echo "Error: Invalid \$1 needs to be an index file"
        exit 1
    fi
    INDEX_FILE="$1"
    shift
    echo ">train.py array-index "${INDEX_FILE}" "${SLURM_ARRAY_TASK_ID}" $*"
    uv run code/src/scripts/train/train.py array-index "${INDEX_FILE}" "${SLURM_ARRAY_TASK_ID}" $* \
        && echo "training completed succesfully."
else
    echo "Not running as a job array"
    echo ">train.py single $*"
    uv run code/src/scripts/train/train.py single $* \
        && echo "training completed succesfully."
fi
echo "---END OF TRAIN SCRIPT--- $(date)"



cd  /lustre/fswork/projects/rech/hhb/commun/lexical-benchmark
# For existing directories recursively: rwx for user & group, nothing for others
find . -type d -exec setfacl --set=u::rwx,g::rwx,g:hhb:rwx,o::---,m::rwx {} \;
# For existing files recursively: rw for user & group, nothing for others
find . -type f -exec setfacl --set=u::rw-,g::rw-,g:hhb:rw-,o::---,m::rw- {} \;


