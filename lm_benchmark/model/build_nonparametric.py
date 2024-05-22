"""build non-parametric model from the train corpus"""
import argparse
import sys
from lm_benchmark.settings import ROOT
from pathlib import Path
from lm_benchmark.utils import TokenCount


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", default=f'{ROOT}/datasets/processed/freq/400h.csv')
    parser.add_argument("--target_file", default=f'{ROOT}/datasets/processed/generation/400.csv')
    return parser.parse_args()



def get_alpha():



    return alpha



def main():
    """ Run the GoldReference loader and write results to a file """
    args = arguments()
    # load the TC object

    make_crp(ref_count: TokenCount, alpha: float)

    target = Path(args.target_file)
    src_file = args.src_file
    header = args.header     # only when loading csv file


    count_df.df.to_csv(target, index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    main()
