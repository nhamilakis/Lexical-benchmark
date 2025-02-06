#!/home/nhamilakis/envs/venvs/lbenchmark/bin/python3.11
# fmt: off
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --job-name=asr-sampler
#SBATCH --time=1:00:00
#SBATCH --export=ALL
#SBATCH --output testloading-%J.log
# fmt: on

import time

from lexical_benchmark.utils import timed_status

with timed_status(status="Loading for X amount of time", complete_status="Completed Loading stuff"):
    time.sleep(20)
