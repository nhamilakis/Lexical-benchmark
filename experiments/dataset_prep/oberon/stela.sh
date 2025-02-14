#!/bin/bash
#SBATCH --job-name=stela-prep
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.log

CODE="${CODE:-$(pwd)/source}"


# Order of operations
#1. --skip_prep
#2. --skip_clean
#3. --skip_fix_lines
#4. --skip_by_month
#5. --skip_by_month_hf_tokenization
#6. --skip_frequency_build
HARDCODED_ARGS=(
    "--skip_prep" "--skip_clean" "--skip_fix_lines" "--skip_frequency_build"
    "--log_level" "DEBUG"
)

if [[ "$*" == *"--debug"* ]]; then
    uv run $CODE/src/scripts/data_prep/stela.py --help
else
    uv run $CODE/src/scripts/data_prep/stela.py "${HARDCODED_ARGS[@]}" "$@"
fi