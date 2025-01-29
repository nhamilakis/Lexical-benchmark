import argparse
import typing as t
from pathlib import Path

import pandas as pd

from lexical_benchmark import settings


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate model training array script")
    parser.add_argument(
        "-o","--output_path",
        type=Path,
        default=Path("/scratch1/projects/lexical-benchmark/v2/datasets/script_arg"),
        help="Output directory for path file",
    )
    parser.add_argument("--model", type=str, default="LSTM", help="Target model name")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists")
    parser.add_argument("--lang", type=str, default="EN", help="Language to test")
    parser.add_argument("--max_num", type=int, default=2, help="Maximum number of chunks to train (0 for all)")
    return parser.parse_args()


def get_chunk_number(chunk_dir: Path) -> int:
    try:
        return int(chunk_dir.name)
    except ValueError:
        return 0


def get_untrained_paths(
    dataset_root: Path, model_root: Path, model_name: str, lang: str, resume: bool, max_num: int = 0
) -> list[tuple[str, str, str, Path]]:
    untrained_paths: list[tuple[str, str, str, Path]] = []

    # Iterate through all datasets
    for dataset_dir in dataset_root.iterdir():
        if not dataset_dir.is_dir():
            continue

        # Get dev path
        dev_path = dataset_dir / "dev" / lang / "char_hf.txt"
        if not dev_path.exists():
            continue

        # Check monthly data
        monthly_path = dataset_dir / "by_month" / lang
        if not monthly_path.exists():
            continue

        # Process each month
        for month_dir in monthly_path.iterdir():
            if not month_dir.is_dir():
                continue

            month_chunks: list[tuple[str, str, str, Path]] = []

            # Get all chunks and sort them by number
            chunk_dirs = sorted([d for d in month_dir.iterdir() if d.is_dir()], key=get_chunk_number)

            # Process each chunk
            for chunk_dir in chunk_dirs:
                # Convert dataset path to model path
                model_path = Path(str(chunk_dir).replace("datasets", "models")) / model_name

                # Check if model needs training
                needs_training = not model_path.exists() or (resume and not (model_path / "training_args.bin").exists())

                if needs_training:
                    month_chunks.append(
                        (dataset_dir.name, month_dir.name, chunk_dir.name, dev_path.relative_to(dataset_root))
                    )

            # Apply max_num limit per month if specified
            month_chunks = month_chunks[:max_num] if len(month_chunks) > max_num else month_chunks
            untrained_paths.extend(month_chunks)

    return untrained_paths


def main() -> None:
    """Main function to generate training paths."""
    args = parse_args()

    # Set up paths
    dataset_root: Path = settings.PATH.dataset_root
    model_root: Path = settings.PATH.DATA_DIR / "models"


    # Get untrained paths
    untrained = get_untrained_paths(
        dataset_root=dataset_root,
        model_root=model_root,
        model_name=args.model,
        lang=args.lang,
        resume=args.resume,
        max_num=args.max_num,
    )

    if untrained:
        # Create output dataframe and save
        filename = f"{args.model}_train-args.index"
        df = pd.DataFrame(untrained)
        df.to_csv(args.output_path / filename, index=False, header=False, sep=" ")
        print(f"Wrote {len(untrained)} paths to {args.output_path}/{filename}")
    else:
        print("No untrained models found")


if __name__ == "__main__":
    main()
