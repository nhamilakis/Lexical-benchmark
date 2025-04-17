#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=%x-%j.log             # Standard output and error log

python src/scripts/analysis/match_freq.py -s 4 --nbins 48
python src/scripts/analysis/compute_metric.py -g gen/v2 -s 4