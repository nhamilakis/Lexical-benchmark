#!/usr/bin/env python
import argparse

from lexical_benchmark import settings
from lexical_benchmark.datasets.gen_child.preparation import GenerationMerger


def parseargs():
    # Run parameters
    parser = argparse.ArgumentParser(description="Organize generations by month")
    parser.add_argument("-g", "--GenPath", type=str, default="gen/v2", help="Generation root directory")
    parser.add_argument("-e", "--hour_per_year", default=1000, type=int, help="Estimated yearly exposure hours")
    return parser.parse_args()


def main() -> None:
    """Build Generation."""
    # Args parser
    args = parseargs()

    model_out_dir = settings.PATH.DATA_DIR / args.GenPath
    print(f"Merging and reorganizing files from {model_out_dir}")
    model_processor = GenerationMerger(hour_per_year=args.hour_per_year, gen_dir=model_out_dir)
    model_processor.process()

    print("Finished merging and saving generations by months.")


if __name__ == "__main__":
    main()
