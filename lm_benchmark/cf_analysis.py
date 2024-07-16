"""Catastrophic forgetting analysis."""

import argparse
from pathlib import Path

import pandas as pd

from lm_benchmark import nlp_tools, settings

from .datasets.machine_cdi import probe_util


def arguments() -> argparse.Namespace:
    """Build & Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Select probe set that is unique to the target dir")
    parser.add_argument(
        "--input_dir", type=str, default=f"{settings.PATH.DATA_DIR / 'datasets/raw/100' }", help="batch dir"
    )
    parser.add_argument(
        "--CDI_dir",
        type=str,
        default=f"{settings.PATH.DATA_DIR / 'datasets/processed/CDI/100' }",
        help="unique probe set dir",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{settings.PATH.DATA_DIR / 'datasets/processed/CF/100.csv'}",
        help="dir to save stat file",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        default=f"{settings.PATH.DATA_DIR / 'datasets/processed/generation/100' }",
        help="generation dir",
    )
    parser.add_argument("--prop_lst", type=list, default=[0.5, 1], help="prop of reserved words")
    parser.add_argument("--run_stat", default=False, help="whether to perform stat")
    return parser.parse_args()


def analyze_pipeline(
    filenames: list[str],
    input_dir: Path,
    gen_dir: Path,
    cdi_dir: Path,
    prop: float,
    gen_files_all: dict,
    probe_files_all: dict,
    *,
    run_stat: bool,
) -> tuple[pd.DataFrame, dict[str, nlp_tools.TokenCount], dict[str, nlp_tools.TokenCount]]:
    """Smallest unit to get stat results."""
    # load files
    print(f"Loading reference files from {input_dir}")
    files = probe_util.load_files(filenames, input_dir, "txt")
    print(f"Loading generation files from {gen_dir}")
    gen_files = probe_util.load_files(filenames, gen_dir, "csv")
    gen_files_all.update(gen_files)
    # build probe tests
    probe_files = probe_util.select_probe_set(files, cdi_dir, prop)
    probe_files_all.update(probe_files)
    print(f"Saved the selected probing set to {cdi_dir}")
    # run stat analysis
    result = probe_util.compare_scores(probe_files, gen_files, run_stat=run_stat)
    return result, gen_files_all, probe_files_all


def main() -> None:
    """Main function to perform CF analysis via CMD."""
    args = arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    cdi_dir = Path(args.CDI_dir)
    gen_path = Path(args.gen_dir)
    run_stat = args.run_stat
    result_all = pd.DataFrame()
    # loop over different epochs
    for prop in args.prop_lst:
        for gen_dir in gen_path.iterdir():
            if gen_dir.is_dir():
                # loop over different temperatures;
                probe_files_all: dict = {}
                gen_files_all: dict = {}

                # rename the generation files
                epoch_dict = probe_util.rename_files(gen_dir)
                for epoch, filenames in epoch_dict.items():
                    result, gen_files_all, probe_files_all = analyze_pipeline(
                        filenames,
                        input_dir,
                        gen_dir,
                        cdi_dir,
                        prop,
                        gen_files_all,
                        probe_files_all,
                        run_stat=run_stat,
                    )
                    print(gen_dir)
                    result["epoch"] = epoch
                    result["temp"] = gen_dir.name
                    result["prop"] = prop
                    result_all = pd.concat([result_all, result])
                if run_stat:
                    # also save the result to the whole
                    result = probe_util.compare_scores(probe_files_all, gen_files_all)
                    result["epoch"] = "all_epoch"
                    result["temp"] = gen_dir.name
                    result["prop"] = prop
                    result_all = pd.concat([result_all, result])

    result_all.to_csv(output_dir)
    print("Finished stat analysis")
