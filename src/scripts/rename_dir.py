#!/usr/bin/env python
"""Rename the month-based convention into chunk-based"""

import argparse
from pathlib import Path

from lexical_benchmark.settings import PATH, month2chunk


def parseargs():
    # Run parameters
    parser = argparse.ArgumentParser(description="rename folders")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="rename_test/by_month/EN",
        help="relative path to the root dir"
        )
    return parser.parse_args()




def main():
    # Args parser
    args = parseargs()

    root_path = PATH.DATA_DIR / args.source_dir
    overlap_lst = [15,30]
    replaced_lst = []
    for month_num in overlap_lst:
        if (root_path/str(month_num)).exists():
            new_dir = root_path / f"{month2chunk(month_num, 1000):02d}"
            replaced_lst.append(new_dir)
            (root_path/str(month_num)).rename(new_dir)
            print(f"replacing the source name {root_path}/{month_num} to {new_dir}")

    for month in root_path.iterdir():
        month_num = int(month.name)
        if month_num > 5 and month not in replaced_lst:
            new_dir = root_path / f"{month2chunk(month_num, 1000):02d}"
            Path(month).rename(new_dir)
            print(f"replacing the source name {month} to {new_dir}")
        elif month_num < 6 and month not in replaced_lst:
            new_dir = root_path / f"{int(month.name):02d}"
            Path(month).rename(new_dir)
            print(f"replacing the source name {month} to {new_dir}")


if __name__ == "__main__":
    main()
