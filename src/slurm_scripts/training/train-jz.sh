#!/bin/bash
#SBATCH --job-name=lb-training
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

uv run code/src/scripts/train/train.py single stela EN 60 00 $1 --resume