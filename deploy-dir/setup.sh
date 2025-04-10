#!/bin/bash

mkdir -p logs
cp -f code/deploy-dir/pyproject.toml . 
cp -f code/deploy-dir/uv.lock .
uv sync --no-build-isolation
