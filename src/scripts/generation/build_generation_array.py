#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd

from lexical_benchmark import settings

# NOTE: these models do not produce output during 20h of generation
# NOTE: there is a bug they should not be add to index while the bug has not been solved
DO_NOT_RUN = {
    "ChildRealistic/by_month/EN/02/00/trans",
    "ChildRealistic/by_month/EN/02/01/trans",
    "ChildRealistic/by_month/EN/03/00/trans",
    "ChildRealistic/by_month/EN/03/01/trans",
    "ChildRealistic/by_month/EN/04/00/trans",
    "ChildRealistic/by_month/EN/04/01/trans",
    "ChildRealistic/by_month/EN/05/00/trans",
    "ChildRealistic/by_month/EN/05/01/trans",
    "ChildRealistic/by_month/EN/06/01/trans",
    "ChildRealistic/by_month/EN/15/00/trans",
    "ChildRealistic/by_month/EN/15/01/trans",
    "ChildRealistic/by_month/EN/25/00/trans",
    "ChildRealistic/by_month/EN/25/01/trans"
}

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Get array script for generation")
    parser.add_argument("-g","--gen-path", type=str, default="gen/merged", help="Generation directory")
    parser.add_argument("-m","--model-path", type=str, default="models", help="Model directory")
    parser.add_argument("-o","--output-path", type=str, default="generation-args.index", help="Output file path")
    parser.add_argument("-e","--hour-per-year", type=int, default=1000, help="Yearly exposure hours")
    parser.add_argument("--override", action="store_true", help="Override previous generation")
    parser.add_argument(
        "--target_months", type=int, nargs="+", default=[6, 12, 18, 24, 30, 36], help="Target months for generation"
    )
    parser.add_argument("--max_num", type=int, default=2, help="Max chunks per month")
    parser.add_argument("--lang", type=str, default="EN", help="Language to test")
    return parser.parse_args()



def get_chunk_number(chunk_dir: Path) -> int:
    """Extract chunk number from directory name."""
    try:
        return int(chunk_dir.name)
    except ValueError:
        return 0


def is_target_month(chunk_num: int, target_months: list[int], hour_per_year: int) -> bool:
    """Check if chunk corresponds to a target month."""
    chunk_month = settings.chunk2month(chunk_num, hour_per_year)
    return chunk_month in target_months


def has_model_been_trained(model_root: Path, dataset: str, lang: str, month: str, chunk: str, model_type: str) -> bool:
    """Check if a specific model has been trained."""
    return (model_root / dataset / "by_month" / lang / month / chunk / model_type / "training_args.bin").is_file()


def has_generation(gen_root: Path, dataset: str, lang: str, month: str, chunk: str, model_type: str, hour_per_year: str) -> bool:
    """Check if model has been generated."""
    return (gen_root / dataset / "by_month" / lang / month / chunk / model_type / f"{hour_per_year}_hour_per_year.csv").is_file()

def has_intermidiate(gen_root: Path, dataset: str, lang: str, month: str, chunk: str, model_type: str, hour_per_year: str) -> bool:
    return (gen_root / dataset / "by_month" / lang / month / chunk / model_type / f"{hour_per_year}_hour_per_year.intermediate.csv").is_file()


def collect_generation_paths(
    root_model_dir: Path, gen_root: Path, target_months: list[int], hour_per_year: int, lang: str, max_num: int, override: bool
) -> tuple[list[Path], list[Path]]:
    """Collect paths for model generation."""
    data_dirs: list[Path] = []
    model_dirs: list[Path] = []

    # Process each dataset directory
    for dataset_name in ("ChildRealistic", "STELATranscriptions2"):
        dataset_dir = root_model_dir / dataset_name
        if not dataset_dir.is_dir():
            continue

        # Check monthly directory
        monthly_path = dataset_dir / "by_month" / lang
        # Process each month directory
        for month_dir in monthly_path.iterdir():
            current_month = month_dir.name
            if not month_dir.is_dir() or not is_target_month(int(month_dir.name), target_months, hour_per_year):
                continue

            # Get and sort chunks
            chunks = sorted([d for d in month_dir.iterdir() if d.is_dir()], key=get_chunk_number)

            # Filter chunks by target months and max_num
            valid_chunks = chunks[:max_num] if len(chunks) > max_num else chunks

            # Process each valid chunk
            for chunk_dir in valid_chunks:
                current_chunk = chunk_dir.name
                # Process each model in chunk
                for model_dir in chunk_dir.iterdir():
                    if not model_dir.is_dir():
                        continue

                    current_model = model_dir.name

                    # Check if model is trained
                    if not has_model_been_trained(root_model_dir, dataset_name, lang, current_month, current_chunk, current_model):
                        print(f"Skip untrained model: {model_dir}")
                        continue

                    # Check if generation needed
                    has_gen = has_generation(gen_root, dataset_name, lang, current_month, current_chunk, current_model, hour_per_year)
                    has_inter = has_intermidiate(gen_root, dataset_name, lang, current_month, current_chunk, current_model, hour_per_year)

                    if (not has_inter and not has_gen) or override:
                        arg_path = Path(dataset_name) / "by_month" / lang / current_month / current_chunk /current_model

                        if str(arg_path) in DO_NOT_RUN:
                            print(f"Skipping DO NOT RUN {arg_path} !")
                        else:
                            data_dirs.append(arg_path)
                            model_dirs.append(arg_path)
                    else:
                        print(f"Generation exists: {Path(dataset_name) / 'by_month' / lang / current_month / current_chunk /current_model}")

    return data_dirs, model_dirs


def main() -> None:
    """Main function."""
    args = parse_args()

    # Setup paths
    root_model_dir = settings.PATH.DATA_DIR / args.model_path
    gen_root = settings.PATH.DATA_DIR / args.gen_path
    output_path = Path.cwd() / args.output_path

    # Collect paths
    data_dirs, model_dirs = collect_generation_paths(
        root_model_dir=root_model_dir,
        gen_root=gen_root,
        target_months=args.target_months,
        hour_per_year=args.hour_per_year,
        lang=args.lang,
        max_num=args.max_num,
        override=args.override,
    )

    if data_dirs:
        # Save results
        df = pd.DataFrame([data_dirs, model_dirs]).T
        df.to_csv(output_path, index=False, header=False, sep=" ")
        print(f"Wrote {len(data_dirs)} paths to {output_path}")
    else:
        print("No paths found matching criteria")


if __name__ == "__main__":
    main()
