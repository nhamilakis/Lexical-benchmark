import argparse
import sys
from pathlib import Path

from lm_benchmark.settings import ROOT
from lm_benchmark.utils import TokenCount


def arguments() -> argparse.Namespace:
    """Build & Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", default=f"{ROOT}/datasets/raw/10_3.txt")
    parser.add_argument("--target_file", default=f"{ROOT}/datasets/processed/freq/10_3.csv")
    parser.add_argument("--header", default="unprompted_1.5")
    return parser.parse_args()


def main() -> None:
    """Run the GoldReference loader and write results to a file."""
    args = arguments()

    target = Path(args.target_file)
    src_file = args.src_file
    header = args.header  # only when loading csv file
    print(f'Loading text from {src_file}')

    if src_file.endswith("txt"):
        count_df = TokenCount.from_text_file(src_file)
    elif src_file.endswith("csv"):
        count_df = TokenCount.from_df(src_file, header)

    count_df.df.to_csv(target, index=False)
    print(f'Writing freq file to {target}')

if __name__ == "__main__":
    args = sys.argv[1:]
    main()
