"""Build non-parametric model from the train corpus."""

import argparse

import pandas as pd

from lm_benchmark import settings
from lm_benchmark.model import model_util


def arguments() -> argparse.Namespace:
    """Build & Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", default=f"{settings.PATH.DATA_DIR}/datasets/processed/freq/800h.csv")
    parser.add_argument("--target_file", default=f"{settings.PATH.DATA_DIR}/datasets/processed/generation/800h.csv")
    parser.add_argument("--fixed_alpha", default=True)
    parser.add_argument("--alpha", type=int, default=80000, help="?")
    parser.add_argument(
        "--desired_oov",
        type=float,
        default=0.023,
        help="Desired OOV rate as a fraction (e.g., 0.05 for 5%).",
    )
    return parser.parse_args()


def calculate_alpha(total_token_count: int, desired_oov: float) -> float:
    """Calculate alpha based on the desired Out-Of-Vocabulary (OOV) rate.

    Parameters
    ----------
    total_token_count:  int
        Total number of tokens in the corpus.
    desired_oov: float
        Desired OOV rate as a fraction (e.g., 0.05 for 5%).

    Returns
    -------
    float
        The calculated value of alpha.

    """
    return desired_oov * total_token_count


def main() -> None:
    """Main Function Allowing calling from CMD."""
    args = arguments()
    ref_count = pd.read_csv(args.src_file).dropna()
    if args.fixed_alpha:
        alpha = args.alpha
        print("Using fixed alpha parameter")
    else:
        alpha = calculate_alpha(ref_count["count"].sum(), args.desired_oov)
        print("Adjusting alpha to reach the desired oov rate")

    gen_count = model_util.make_crp(ref_count, alpha)
    gen_count.to_csv(args.target_file, index=False)


if __name__ == "__main__":
    main()
