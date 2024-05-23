"""build non-parametric model from the train corpus"""
import argparse
import sys
import pandas as pd
from lm_benchmark.settings import ROOT
from lm_benchmark.utils import TokenCount
from model_util import make_crp

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", default=f'{ROOT}/datasets/processed/freq/800h.csv')
    parser.add_argument("--target_file", default=f'{ROOT}/datasets/processed/generation/800.csv')
    parser.add_argument("--fixed_alpha", default=True)
    parser.add_argument("--alpha", default=80000)
    parser.add_argument("--desired_oov", default=0.023)
    return parser.parse_args()



def calculate_alpha(total_token_count: int, desired_oov: float) -> float:
    """
    Calculate alpha based on the desired Out-Of-Vocabulary (OOV) rate.
    Args:
        total_token_count (int): Total number of tokens in the corpus.
        desired_oov_rate (float): Desired OOV rate as a fraction (e.g., 0.05 for 5%).

    Returns:
        float: The calculated value of alpha.
    """
    return desired_oov * total_token_count



def main():

    args = arguments()

    # load the TC object
    ref_count = TokenCount()
    ref_count.df = pd.read_csv(args.src_file).dropna()

    if args.fixed_alpha:
        alpha = args.alpha
    else:
        alpha = calculate_alpha(ref_count.df, args.desired_oov)

    gen_count = make_crp(ref_count, alpha)
    gen_count.to_csv(args.target_file, index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    main()
