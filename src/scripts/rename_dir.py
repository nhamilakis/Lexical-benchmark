"""rename the month-based convention into chunk-based"""

import argparse
from pathlib import Path

from lexical_benchmark.settings import PATH, month2chunk


def parseargs():
    # Run parameters
    parser = argparse.ArgumentParser(description="rename folders")
    parser.add_argument("--source_dir", type=str, default="rename_test/by_month/EN", help="Source Directory1")
    return parser.parse_args()


def main():
    # Args parser
    args = parseargs()

    root_path = PATH.DATA_DIR / args.source_dir

    # convert 15th month to the chunk first to avoid overlapping
    new_dir_15 = root_path / f"{month2chunk(15, 1000):02d}"
    Path(root_path / "15").rename(new_dir_15)
    print(f"replacing the source name {root_path}/15 to {new_dir_15}")

    for month in root_path.iterdir():
        month_num = int(month.name)

        if month_num > 5 and month != new_dir_15.name:
            new_dir = root_path / f"{month2chunk(month_num, 1000):02d}"
        else:
            new_dir = root_path / f"{int(month.name):02d}"

        Path(month).rename(new_dir)
        print(f"replacing the source name {root_path / 15} to {new_dir_15}")


if __name__ == "__main__":
    main()
