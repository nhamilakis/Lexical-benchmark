#!/home/nhamilakis/envs/venvs/lbenchmark/bin/python3.11
# fmt: off
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=childes-cleanups
#SBATCH --time=3:00:00
#SBATCH --export=ALL
#SBATCH --output childes-clean-%J.log
# fmt: on
import os
from pathlib import Path

import pandas as pd
from tap import Tap

from lexical_benchmark.datasets import childes
from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.utils import slurm_utils

slurm_utils.info_header()


class STELACleanArgs(Tap):
    """CMD args for STELA Cleanup PIPELINE."""

    location: str
    extras_id: str | None = None  # CHILDES Extra dict
    save_args: bool = True  # Save arguments
    skip_formatting: bool = False  # Skip Dataset Pre-Formatting
    skip_cleaning: bool = False  # Skip Dataset Cleaning
    skip_word_cleaning: bool = False  # Skip Dataset Word Cleaning
    skip_word_frequencies: bool = False  # Skip Extraction of Word Frequencies

    def configure(self) -> None:
        """Extra configuration."""
        self.add_argument("location")

    def cache_args(self) -> None:
        """Save Arguments to disk."""
        if args.save_args:
            Path("arg_cache").mkdir(exist_ok=True)
            slurm_id = ""
            if "SLURM_JOB_ID" in os.environ:
                slurm_id = "_" + os.environ["SLURM_JOB_ID"]
            self.save(f"cache/args_asr{slurm_id}.json")


## Arguments setup
args_loader = STELACleanArgs()
if "ARGS" in os.environ:
    arg_file = Path(os.environ["ARGS"])
    args: STELACleanArgs = args_loader.from_dict(arg_file.load_json())
else:
    args = args_loader.parse_args()

slurm_utils.info_args(args)
args.cache_args()

prog_file = Path.cwd() / "childes.progress"
root_dir = Path(args.location)
dataset = childes.CHILDESDataset(root_dir=root_dir)

if not args.skip_formatting:
    progress = slurm_utils.ProgressTask(task_name="childes_prep", update_interval=20, target_file=prog_file)
    prep = childes.CHILDESPreparation()

    # Add all subsets
    for accent in progress.sequence_progress(dataset.accents):
        prep.load_dir(dataset.source_path / accent, accent)

    # Extract data
    with progress.parallel_progress("Extracting from CHILDES"):
        prep.export(dataset.preprocessed_path, meta_dir=dataset.root_dir / "metadata")
        prep.export_turn_taking(dataset.preprocessed_path)

    progress.complete()

    # Build ID Mapping
    progress = slurm_utils.ProgressTask(task_name="childes_id", update_interval=10, target_file=prog_file)
    for accent in progress.sequence_progress(dataset.accents):
        current = dataset.source_path / accent
        id_list = [(item.relative_to(current).parent / item.stem).parts for item in root_dir.rglob("*.cha")]
        (dataset.root_dir / "metadata" / f"ids_{accent}.txt").write_text(
            "\n".join([",".join(parts) for parts in id_list])
        )
    progress.complete()
else:
    print("Skipping Dataset preFormatting...", flush=True)


if not args.skip_cleaning:
    progress = slurm_utils.ProgressTask(task_name="childes_clean", update_interval=30, target_file=prog_file)
    # Text cleaning & Tag Extraction
    with progress.parallel_progress("CHILD"):
        dataset_utils.DatasetCleaner.cleanup_files(
            filemap=dataset.raw2processed_filesmap("child"), ruleset=dataset.clean_rulespec("child"), save_logs=True
        )
    with progress.parallel_progress("ADULT"):
        dataset_utils.DatasetCleaner.cleanup_files(
            filemap=dataset.raw2processed_filesmap("adult"), ruleset=dataset.clean_rulespec("adult"), save_logs=True
        )

    # Build extras dictionairy from tags
    with progress.parallel_progress("TAGS"):
        childes_adult_extras_lexique = childes.CHILDESExtrasLexicon(dataset)
        childes_adult_extras_lexique.add_lang("Eng-NA", "adult")
        childes_adult_extras_lexique.add_lang("Eng-UK", "adult")
        dict_hash_id = childes_adult_extras_lexique.cache_current()

    args.extras_id = dict_hash_id
    args.cache_args()
    progress.complete()
else:
    print("Skipping Text Cleaning...", flush=True)


if not args.skip_word_cleaning:
    # Load full dictionairy
    childes_word_cleaner = dataset_utils.DictionairyCleaner(lang="EN", childes_extra_id=args.extras_id)
    progress = slurm_utils.ProgressTask(task_name="childes_word_clean", update_interval=30, target_file=prog_file)
    with progress.parallel_progress("ADULT"):
        dataset_utils.DatasetCleaner.word_validate_files(
            filemap=dataset.word_validation_filesmap("adult"),
            cleaner=childes_word_cleaner,
        )

    with progress.parallel_progress("ADULT"):
        dataset_utils.DatasetCleaner.word_validate_files(
            filemap=dataset.word_validation_filesmap("child"),
            cleaner=childes_word_cleaner,
        )
    progress.complete()
else:
    print("Skipping Word Cleaning...", flush=True)


if not args.skip_word_frequencies:
    # Word Frequencies
    progress = slurm_utils.ProgressTask(task_name="childes_word_freqs", update_interval=30, target_file=prog_file)
    wf_index = pd.Index(["word", "freq"])

    for lang_accent in dataset.accents:
        for speech_type in dataset.speech_types:
            dataset.wf.rejected(lang_accent, speech_type).mk_parent()  # Create parent dir

            # Clean
            clean_word_frequencies = dataset.build_word_frequencies(
                lang_accent=lang_accent, speech_type=speech_type, word_type="clean"
            )
            df = pd.DataFrame.from_dict(clean_word_frequencies, orient="index").reset_index()
            df.columns = wf_index
            df.to_csv(dataset.wf.clean(lang_accent, speech_type), index=False)

            # Rejected
            rejected_word_frequencies = dataset.build_word_frequencies(
                lang_accent=lang_accent, speech_type=speech_type, word_type="rejected"
            )
            df = pd.DataFrame.from_dict(rejected_word_frequencies, orient="index").reset_index()
            df.columns = wf_index
            df.to_csv(dataset.wf.rejected(lang_accent, speech_type), index=False)

            # Processed
            processed_word_frequencies = dataset.build_word_frequencies(
                lang_accent=lang_accent, speech_type=speech_type, word_type="processed"
            )
            df = pd.DataFrame.from_dict(processed_word_frequencies, orient="index").reset_index()
            df.columns = wf_index
            df.to_csv(dataset.wf.processed(lang_accent, speech_type), index=False)
    progress.complete()
else:
    print("Skipping Word-Frequency build...")

slurm_utils.info_footer()
