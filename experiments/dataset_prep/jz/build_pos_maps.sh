#!/bin/bash
#SBATCH --job-name=build-pos-maps
#SBATCH --account=hhb@cpu
# Partition (CPU)
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
# Only run this when testing
##SBATCH --qos=qos_cpu-dev
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j-%a.log
# hyperthreading is deactivated
#SBATCH --hint=nomultithread

CODE="${CODE:-$(pwd)/code}"

HARDCODED_ARGS=(
    "--lang" "EN" --no_build_pos
    "--spacy_pos_model" "en_core_web_trf" "--spacy_batch_size" "2048"
)

if [[ "$*" == *"--debug"* ]]; then
    uv run $CODE/src/scripts/dataset_clean/build_pos_map.py --help
else
    uv run $CODE/src/scripts/dataset_clean/build_pos_map.py "${HARDCODED_ARGS[@]}" "$@"
fi