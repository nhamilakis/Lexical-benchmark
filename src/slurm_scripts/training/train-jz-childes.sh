#!/bin/bash
#SBATCH --job-name=training
#SBATCH --account=hhb@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-1  # Adjust this range depending on the number of tasks
#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --hint=nomultithread

echo "---START OF TRAIN SCRIPT--- $(date)"
export CLUSTER_NAME="jean-zay"

# Define an array of commands
commands=(
  "uv run src/scripts/train/train.py single childes_adult EN 0 0 lstm --batch-size 256 --resume"
  "uv run src/scripts/train/train.py single childes_adult EN 0 0 gpt2 --batch-size 32 --resume"
)

# Execute the command corresponding to the SLURM_ARRAY_TASK_ID
${commands[$SLURM_ARRAY_TASK_ID]}

echo "---END OF TRAIN SCRIPT--- $(date)"
