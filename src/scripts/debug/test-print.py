#!.venv/bin/python
# fmt: off
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --job-name=test-logs
#SBATCH --time=0:05:00
#SBATCH --export=ALL
#SBATCH --output test-%J.log
# fmt: on

import os

from lexical_benchmark.utils import slurm_utils

slurm_utils.info_header()

if os.environ.get("SLURM_JOB_ID") is None:
    print("Why is JOB ID None ? when in slurm ?")
print("HELLO???")
# Job Done
slurm_utils.info_footer()
