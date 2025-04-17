#!/bin/bash
#SBATCH --job-name=stela-prep
#SBATCH --account=hhb@cpu
# Partition (A100)
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
# Only run this when testing
##SBATCH --qos=qos_cpu-dev
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j-%a.log
# hyperthreading is deactivated
#SBATCH --hint=nomultithread

CODE="${CODE:-$(pwd)/code}"

HARDCODED_ARGS=(
    "--skip_word_clean" "--skip_frequency_build"
    "/lustre/fswork/projects/rech/hhb/ucx81cx/data2/datasets/STELATranscriptions"
)

if [[ "$*" == *"--debug"* ]]; then
    uv run $CODE/src/scripts/dataset_clean/stela.py --help
else
    uv run $CODE/src/scripts/dataset_clean/stela.py "${HARDCODED_ARGS[@]}" "$@"
fi