#!/lustre/fsn1/projects/rech/hhb/ucx81cx/conda/lm_dev/bin/python
# fmt: off
#SBATCH --job-name=wordstats-prep
#SBATCH --account=hhb@a100
#SBATCH -C a100                 # Partition (A100)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Number of GPUs per task (On a100 8 GPUs per node are available.)
#SBATCH --gres=gpu:1
# Number of cores per task for gpu_p5 (1/8 of 8-GPUs A100 node)
# A100 nodes have 64 cores, should use proportional to GPU number (1 gpu 1/8 of the CPUs)
# For 4 GPUs use 32 cores per task
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j-%a.log
#SBATCH --hint=nomultithread    # hyperthreading is deactivated
# Only run this when testing
##SBATCH --qos=qos_gpu_a100-dev
# fmt: on
"""Build script for wordstats dataset."""
import os
from pathlib import Path

from tap import Tap

from lexical_benchmark.datasets.wordstats import WordStatsDataset, preparation
from lexical_benchmark.utils import slurm_utils

slurm_utils.info_header()


class WordStatsPrepArgs(Tap):
    """CMD arguments for prepation of the wordstats dataset."""

    skip_fetch_word_counts: bool = True
    skip_pos_computation: bool = False
    require_gpu: bool = True
    pos_model: str = "en_core_web_trf"
    batch_size: int = 2048
    save_args: bool = False

    def cache_args(self) -> None:
        """Save Arguments to disk."""
        if args.save_args:
            Path("arg_cache").mkdir(exist_ok=True)
            slurm_id = ""
            if "SLURM_JOB_ID" in os.environ:
                slurm_id = "_" + os.environ["SLURM_JOB_ID"]
            self.save(f"cache/args_asr{slurm_id}.json")


## Arguments setup
args_loader = WordStatsPrepArgs()
## Arguments setup
args_loader = WordStatsPrepArgs()
if "ARGS" in os.environ:
    arg_file = Path(os.environ["ARGS"])
    args: WordStatsPrepArgs = args_loader.from_dict(arg_file.load_json())
else:
    args = args_loader.parse_args()

slurm_utils.info_args(args)
args.cache_args()

# Fetch Word-Count CSVs from their respective datasets
if not args.skip_fetch_word_counts:
    preparation.prepare_word_stats_word_counts()
else:
    print("Skipping fetch of word-counts...", flush=True)


if not args.skip_pos_computation:
    # compute pos for all words in the dataset
    print("Extracting POS tags for words...", flush=True)
    word_pos_mapping = preparation.build_word_pos_maps(
        args.pos_model, require_gpu=args.require_gpu, batch_size=args.batch_size
    )
    print("Finished POS Extraction...", flush=True)
    dataset = WordStatsDataset(lang="EN")

    # Attach POS to STELA word-counts
    print("Attaching to STELA", flush=True)
    preparation.attach_pos(dataset.word_frequencies.stela_by_month_60_00, word_pos_mapping)

    # Attach POS to CHILDES/adult word-counts
    print("Attaching to CHILDES", flush=True)
    preparation.attach_pos(dataset.word_frequencies.childes_adult, word_pos_mapping)

    # Attach POS to CHILDRealistic word-counts
    print("Attaching to CHILDRealistic", flush=True)
    preparation.attach_pos(dataset.word_frequencies.child_realistic_by_month_60_00, word_pos_mapping)

    # Attach POS to CDI-CHILDRealistic word-counts
    print("Attaching to CDI-CHILDRealistic", flush=True)
    preparation.attach_pos(dataset.word_frequencies.cdi_childrealistic, word_pos_mapping)

    # Attach POS to CDI-CHILDES word-counts
    print("Attaching to CDI-CHILDES", flush=True)
    preparation.attach_pos(dataset.word_frequencies.cdi_childes, word_pos_mapping)

else:
    print("Skipping POS tagging...", flush=True)


slurm_utils.info_footer()
