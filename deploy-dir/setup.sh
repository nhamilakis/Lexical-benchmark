#!/bin/bash

git clone https://github.com/nhamilakis/Lexical-benchmark code
mkdir -p logs
cp code/deploy-dir/pyproject.toml . 
cp code/deploy-dir/uv.lock .
uv sync
