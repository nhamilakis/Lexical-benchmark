#!/usr/bin/env python
import os
from pathlib import Path

import pandas as pd
from tap import Tap

from lexical_benchmark.datasets import childes
from lexical_benchmark.datasets import ChildRealistic
from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.utils import slurm_utils



class STELACleanArgs(Tap):
    """CMD args for ChildRealistic Cleanup PIPELINE."""
    #TODO: replace it with the root dir
    location: str = '/scratch1/projects/lexical-benchmark/v2/datasets/ChildRealistic/'
    extras_id: str | None = None  # CHILDES Extra dict
    save_args: bool = True  # Save arguments
    skip_cleaning: bool = False  # Skip Dataset Cleaning
    skip_word_cleaning: bool = True  # Skip Dataset Word Cleaning
    skip_word_frequencies: bool = True # Skip Extraction of Word Frequencies
    lang: str = "EN"
    


## Arguments setup
args_loader = STELACleanArgs()
if "ARGS" in os.environ:
    arg_file = Path(os.environ["ARGS"])
    args: STELACleanArgs = args_loader.from_dict(arg_file.load_json())
else:
    args = args_loader.parse_args()




dataset = ChildRealistic.ChildRealDataset(root_dir=Path(args.location))
prog_file = Path.cwd() / "stela.progress"
print('Dataset loaded')


if not args.skip_cleaning:
    # Text cleaning & Tag Extraction

    progress = slurm_utils.ProgressTask(task_name="stela_clean", target_file=prog_file)
    files_iter = progress.iter_progress(dataset.raw2clean_filesmap())
    dataset_utils.DatasetCleaner.cleanup_files(
        filemap=files_iter, ruleset=dataset.clean_up_rules('adult'), save_logs=True
    )

    print('Dataset cleaned')
    # Build extras dictionairy from tags;
    # TODO: note here we need to verify whether it is still necessary
    dataset = childes.CHILDESDataset()
    childes_adult_extras_lexique = childes.CHILDESExtrasLexicon(dataset)
    childes_adult_extras_lexique.add_lang("Eng-NA", "adult")
    childes_adult_extras_lexique.add_lang("Eng-UK", "adult")
    dict_hash_id = childes_adult_extras_lexique.cache_current()
    en_dict = dataset_utils.DictionairyCleaner(lang="EN", childes_extra_id=dict_hash_id)

else:
    print("Skipping Text Cleaning...", flush=True)



# TODO: update other parts
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


