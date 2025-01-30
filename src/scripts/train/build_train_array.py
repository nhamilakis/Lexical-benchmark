#!/usr/bin/env python
import argparse
import collections
from pathlib import Path

import pandas as pd

from lexical_benchmark import settings
from lexical_benchmark.datasets.utils import training_files

TRAIN_CHUNKS = ("00", "01")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate model training array script")
    parser.add_argument(
        "-o","--output_path",
        type=Path,
        default=Path.cwd(),
        help="Output directory for path file",
    )
    parser.add_argument("--model", type=str, default="LSTM", help="Target model name")
    parser.add_argument("--override", action="store_true", help="Resume from checkpoint if exists")
    parser.add_argument("--lang", type=str, default="EN", help="Language to test")
    parser.add_argument("--max_num", type=int, default=2, help="Maximum number of chunks to train (0 for all)")
    return parser.parse_args()


def needs_training(model_dir: Path, *, override: bool) -> bool:
    """Check if a model path requires additional training."""
    if not model_dir.is_dir():
        return True
    if override:
        return True

    return not (model_dir / "training_args.bin").is_file()



def get_untrained_paths(
    dataset_root: Path, model_root: Path, model_name: str, lang: str, *, override: bool, max_num: int = 0
) -> list[tuple[str, str, str, Path]]:
    """Crawl Dataset folders and figure out if models have been trained."""
    untrained_paths: dict[str, list[tuple[str, str, str, Path]]] = collections.defaultdict(list)

    for dataset in ("ChildRealistic", "STELATranscriptions2"):
        if not (dataset_root / dataset).is_dir():
            continue

        dev_path = dataset_root / dataset / "dev" / lang / "char_hf.txt"
        if not dev_path.exists():
            continue


        iter_items = training_files.iter_train_structure(
            root_dir=dataset_root, dataset_name=dataset, lang=lang, model_type=model_name)
        for item in iter_items:
            model_path = item.get_model_path(model_root)
            # Check if model needs training
            if needs_training(model_path, override=override) and item.chunk in TRAIN_CHUNKS:
                untrained_paths[f"{dataset}-{item.month}"].append(
                    (item.dataset, item.month, item.chunk, dev_path.relative_to(dataset_root))
                )


    # Apply max_num limit per month if specified
    final_items = []
    for items in untrained_paths.values():
            current_chunks = sorted(items, key=lambda x: x[2])[:max_num] if max_num > 0 else items
            final_items.extend(current_chunks)
    return final_items


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
        override=args.override,
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
