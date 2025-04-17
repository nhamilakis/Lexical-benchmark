#!/bin/bash
#SBATCH --job-name=build-pos-maps
#SBATCH --time=4:00:00
#######################
## Partition (CPU)
##SBATCH --account=hhb@cpu
##SBATCH --partition=cpu_p1
##SBATCH --cpus-per-task=6
##SBATCH --qos=qos_cpu-dev  # For debug
#######################
## Partition (A100)
#SBATCH --account=hhb@a100
#SBATCH -C a100
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#######################
## Global Settings
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x-%j-%a.log
#SBATCH --hint=nomultithread # no need for hyperthreading

CODE="${CODE:-$(pwd)/code}"

HARDCODED_ARGS=(
    "--lang" "EN" --skip_prep_src
    "--spacy_pos_model" "en_core_web_trf" "--spacy_batch_size" "2048"
)

if [[ "$*" == *"--debug"* ]]; then
    uv run $CODE/src/scripts/data_prep/build_pos_map.py --help
else
    uv run $CODE/src/scripts/data_prep/build_pos_map.py "${HARDCODED_ARGS[@]}" "$@"
fi