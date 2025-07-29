#!/bin/bash
#SBATCH --job-name=ressource-check
#SBATCH --account=hhb@a100
# Partition (A100)
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Number of GPUs per task (On a100 8 GPUs per node are available.)
#SBATCH --gres=gpu:1
# Number of cores per task for gpu_p5 (1/8 of 8-GPUs A100 node)
# A100 nodes have 64 cores, should use proportional to GPU number (1 gpu 1/8 of the CPUs)
# For 4 GPUs use 32 cores per task
#SBATCH --cpus-per-task=4
# Only run this when testing
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=00:05:00
# Array Number of Jobs to run in Parallel
##SBATCH --array=0-32
#SBATCH --output=/lustre/fswork/projects/rech/hhb/ucx81cx/logs/%x-%j.log
#SBATCH --hint=nomultithread            # hyperthreading is deactivated

# All files created belong to the project
umask 007

# ENV setup, if not set
if [[ -z "${_LM_ENV}" ]]; then
    source $WORK/load.sh
fi
# SLURM_ARRAY_TASK_ID

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


echo "computation start $(date)"
# launch your computation


echo -e "\n=== Running Python Hardware Check ==="
python check_hardware.py


echo "computation end : $(date)"
