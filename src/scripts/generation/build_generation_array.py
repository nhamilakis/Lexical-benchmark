import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lexical_benchmark import settings


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Get array script for generation")
    parser.add_argument("--gen_path", type=str, default="gen/merged", help="Generation directory")
    parser.add_argument("--model_path", type=str, default="models", help="Model directory")
    parser.add_argument("--output_path", type=str, default="generation-args.index", help="Output file path")
    parser.add_argument("--hour_per_year", type=int, default=1000, help="Yearly exposure hours")
    parser.add_argument("--resume", action="store_true", help="Resume from previous generation")
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


def collect_generation_paths(
    root_dir: Path, gen_root: Path, target_months: list[int], hour_per_year: int, lang: str, max_num: int, resume: bool
) -> tuple[list[Path], list[Path]]:
    """Collect paths for model generation."""
    data_dirs: list[Path] = []
    model_dirs: list[Path] = []

    # Process each dataset directory
    for dataset_dir in tqdm(root_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        # Check monthly directory
        monthly_path = dataset_dir / "by_month" / lang
        if not monthly_path.exists():
            print(f"Monthly path does not exist: {monthly_path}")
            continue

        # Process each month directory
        for month_dir in monthly_path.iterdir():
            if not month_dir.is_dir():
                continue

            # Get and sort chunks
            chunks = sorted([d for d in month_dir.iterdir() if d.is_dir()], key=get_chunk_number)

            # Filter chunks by target months and max_num
            valid_chunks = []
            for chunk in chunks:
                chunk_num = get_chunk_number(chunk)
                if is_target_month(chunk_num, target_months, hour_per_year):
                    valid_chunks.append(chunk)

                    if max_num > 0 and len(valid_chunks) >= max_num:
                        break

            # Process each valid chunk
            for chunk_dir in valid_chunks:
                # Process each model in chunk
                for model_dir in chunk_dir.iterdir():
                    if not model_dir.is_dir():
                        continue

                    # Check if model is trained
                    if not (model_dir / "training_args.bin").exists():
                        print(f"Skip untrained model: {model_dir}")
                        continue

                    # Get generation path
                    gen_path = Path(str(model_dir).replace(str(root_dir), str(gen_root)))
                    gen_file = gen_path / f"{hour_per_year}_hour_per_year.csv"

                    # Check if generation needed
                    if not resume or not gen_file.exists():
                        data_dirs.append(model_dir.relative_to(root_dir))
                        model_dirs.append(gen_path.relative_to(gen_root))
                    else:
                        print(f"Generation exists: {gen_file}")

    return data_dirs, model_dirs


def main() -> None:
    """Main function."""
    args = parse_args()

    # Setup paths
    root_dir = settings.PATH.DATA_DIR / args.model_path
    gen_root = settings.PATH.DATA_DIR / args.gen_path
    output_path = Path.cwd() / args.output_path

    # Collect paths
    data_dirs, model_dirs = collect_generation_paths(
        root_dir=root_dir,
        gen_root=gen_root,
        target_months=args.target_months,
        hour_per_year=args.hour_per_year,
        lang=args.lang,
        max_num=args.max_num,
        resume=args.resume,
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
