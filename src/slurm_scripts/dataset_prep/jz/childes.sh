#!/bin/bash
#SBATCH --job-name=childes-prep
#SBATCH --account=hhb@cpu
# Partition (A100)
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
# Only run this when testing
##SBATCH --qos=qos_cpu-dev
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j-%a.log
# hyperthreading is deactivated
#SBATCH --hint=nomultithread

CODE="${CODE:-$(pwd)/code}"

HARDCODED_ARGS=(
    "--skip_word_cleaning" "--skip_word_frequencies"
    "/lustre/fswork/projects/rech/hhb/ucx81cx/data2/datasets/CHILDES"
)

if [[ "$*" == *"--debug"* ]]; then
    uv run $CODE/src/scripts/dataset_clean/childes.py --help
else
    uv run $CODE/src/scripts/dataset_clean/childes.py "${HARDCODED_ARGS[@]}" "$@"
fi