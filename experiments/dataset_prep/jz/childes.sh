#!/bin/bash
#SBATCH --job-name=childes-prep
#SBATCH --account=hhb@a100
# Partition (A100)
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
# Only run this when testing
##SBATCH --qos=qos_cpu-dev
#SBATCH --time=5:00:00
#SBATCH --output=logs/%x-%j-%a.log
# hyperthreading is deactivated
#SBATCH --hint=nomultithread

CODE="${CODE:-source}"

CUSTOM_ARGS=(
    
)

uv run $CODE/src/scripts/dataset_clean/childes.py $CUSTOM$@