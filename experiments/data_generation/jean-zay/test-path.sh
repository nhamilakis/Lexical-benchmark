#!/bin/bash

export MODEL_ROOT="$WORK/data/models"
export GEN_ROOT="$WORK/data/gen"
export DATASET_ROOT="$WORK/data/datasets"
export CODE="$(pwd)/code"
export JZ=1


echo "PYTHON: $(uv run python -V)"
echo "PYTHON: $(uv run which python)"

echo "CMD:"
uv run $CODE/src/scripts/generation/generate.py --help