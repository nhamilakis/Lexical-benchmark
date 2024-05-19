import argparse
import sys
from lm_benchmark.settings import ROOT
from pathlib import Path
from lm_benchmark.utils import TokenCount

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", default=f'{ROOT}/datasets/processed/generation/generation.csv')
    parser.add_argument("--target_file", default=f'{ROOT}/datasets/processed/freq/unprompted_0.6.csv')
    parser.add_argument("--header",default='unprompted_0.6')
    return parser.parse_args()


def main():
    """ Run the GoldReference loader and write results to a file """
    args = arguments()

    target = Path(args.target_file)
    src_file = args.src_file
    header = args.header     # only when loading csv file
    if src_file.endswith('txt'):
        count_df = TokenCount.from_text_file(src_file)
    elif src_file.endswith('csv'):
        count_df = TokenCount.from_df(src_file,header)

    count_df.df.to_csv(target, index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    main()

