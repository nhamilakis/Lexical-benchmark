#!/home/nhamilakis/envs/venvs/lbenchmark/bin/python3.11
# fmt: off
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=stela-cleanups
#SBATCH --time=5:00:00
#SBATCH --export=ALL
#SBATCH --output stela-clean-%J.log
# fmt: on

import os
from pathlib import Path

from tap import Tap

from lexical_benchmark import settings
from lexical_benchmark.datasets import stella
from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.utils import slurm_utils

slurm_utils.info_header()


class STELACleanArgs(Tap):
    """CMD args for STELA Cleanup PIPELINE."""

    location: str
    lang: str = "EN"
    save_args: bool = True  # Save arguments
    skip_prep: bool = True  # Skip preparation
    skip_clean: bool = False  # Skip text cleaning
    skip_word_clean: bool = False  # Skip word cleaning
    skip_frequency_build: bool = False  # Skip building of Frequency Maps
    asr_location: str = str(settings.PATH.asr_dir)  # Location of ASR text
    bad_books: tuple[str, ...] = (
        "4262_LibriVox_en",
        "6910_LibriVox_en",
        "4955_LibriVox_en",
        "5726_LibriVox_en",
        "5788_LibriVox_en",
    )

    def configure(self) -> None:
        """Extra configuration."""
        self.add_argument("location")


## Arguments setup
args_loader = STELACleanArgs()
if "ARGS" in os.environ:
    arg_file = Path(os.environ["ARGS"])
    args: STELACleanArgs = args_loader.from_dict(arg_file.load_json())
else:
    args = args_loader.parse_args()

slurm_utils.info_args(args)

if args.save_args:
    Path("cache/args").mkdir(exist_ok=True)
    slurm_id = ""
    if "SLURM_JOB_ID" in os.environ:
        slurm_id = "_" + os.environ["SLURM_JOB_ID"]
    args.save(f"cache/args/stela_clean{slurm_id}.json")
prog_file = Path.cwd() / "stela.progress"

if not args.skip_prep:
    # Prepare STELA STRUCTURE
    progress = slurm_utils.ProgressTask(task_name="stela_prep", update_interval=20, target_file=prog_file)
    prep = stella.STELAPrepTranscripts(
        root_dir=Path(args.location),
        lang=args.lang,
        with_asr=Path(args.asr_location) if args.asr_location else None,
        bad_books=args.bad_books,
    )
    with progress.parallel_progress():
        prep.build_transcript()
    progress.complete()
else:
    print("Skipping dataset preparation", flush=True)

dataset = stella.STELATranscriptDataset(root_dir=Path(args.location))

if not args.skip_clean:
    # Cleanup & Normalise text files
    progress = slurm_utils.ProgressTask(task_name="stela_clean", target_file=prog_file)
    files_iter = progress.iter_progress(dataset.raw2clean_filesmap(args.lang))
    dataset_utils.DatasetCleaner.cleanup_files(
        filemap=files_iter, ruleset=dataset.clean_up_rules(args.lang), save_logs=True
    )
    progress.complete()
else:
    print("Skipping text cleaning", flush=True)

if not args.skip_word_clean:
    # Filter words using a dictionairy
    progress = slurm_utils.ProgressTask(task_name="stela_word_validation", target_file=prog_file)
    files_iter = progress.iter_progress(dataset.word_validation_filesmap(args.lang))
    dataset_utils.DatasetCleaner.word_validate_files(
        filemap=files_iter,
        cleaner=dataset_utils.DictionairyCleaner(lang=args.lang),
    )
    progress.complete()
else:
    print("Skipping word cleaning", flush=True)

if not args.skip_frequency_build:
    # Build Frequency Maps
    progress = slurm_utils.ProgressTask(task_name="stela_word_frequencies", target_file=prog_file, update_interval=20)
    with progress.parallel_progress():
        dataset.build_clean_word_frequencies()
        dataset.build_rejected_word_frequencies()
        dataset.build_preprocess_word_frequencies()
    progress.complete()
else:
    print("Skipping word-frequency map build", flush=True)

slurm_utils.info_footer()
