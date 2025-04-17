#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --job-name=pos-tagging
#SBATCH --time=02:00:00
#SBATCH --export=ALL
#SBATCH --output %x-%J.log

CODE="${CODE:-$(pwd)/source}"

HARDCODED_ARGS=(
    "--lang" "EN" --no_build_pos
    "--spacy_pos_model" "en_core_web_trf" "--spacy_batch_size" "2048"
)

if [[ "$*" == *"--debug"* ]]; then
    uv run $CODE/src/scripts/dataset_clean/build_pos_map.py --help
else
    uv run $CODE/src/scripts/dataset_clean/build_pos_map.py "${HARDCODED_ARGS[@]}" "$@"
fi