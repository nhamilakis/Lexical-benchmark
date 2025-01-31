#!/bin/bash

export MODEL_ROOT="$WORK/data/models"
export GEN_ROOT="$WORK/data/gen"
export DATASET_ROOT="$WORK/data/datasets"
export CODE="$(pwd)/code"
export JZ=1

echo "PYTHON: $(uv run python -V)"
echo "PYTHON: $(uv run which python)"

echo "CMD1:"
uv run $CODE/src/scripts/train/hf/train_trans.py --help

echo "CMD2:"
uv run $CODE/src/scripts/train/hf/train_LSTM.py --help