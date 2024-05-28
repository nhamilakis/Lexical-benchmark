"""convert word count into accumulated monthly count."""
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from lm_benchmark.analysis.score_util import MonthCounter
from lm_benchmark.settings import ROOT, AGE_DICT



def arguments() -> argparse.Namespace:
    """Build & Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file", default=f"{ROOT}/datasets/processed/generation/crf.csv")
    parser.add_argument("--est_file", default=f"{ROOT}/datasets/raw/vocal_month.csv")
    parser.add_argument("--CDI_path", default=f"{ROOT}/datasets/processed/CDI/")
    parser.add_argument("--freq_path", default=f"{ROOT}/datasets/processed/month_count/")
    parser.add_argument("--test_type", default="exp")
    parser.add_argument("--lang", default="BE")
    parser.add_argument("--set", default="machine")
    parser.add_argument(
        "--header_lst", default=["unprompted_0.3","unprompted_0.6","unprompted_1.0","unprompted_1.5"]
    )
    parser.add_argument("--count", default=True)
    return parser.parse_args()


def main() -> None:
    """Run the GoldReference loader and write results to a file."""
    args = arguments()
    gen_file = Path(args.gen_file)
    est_file = Path(args.est_file)
    lang = args.lang
    header_lst = args.header_lst
    test_file = Path(args.CDI_path) / f"{lang}_{args.test_type}_{args.set}.csv"
    score_dir = Path(f"{args.freq_path}{lang}")
    score_dir.mkdir(parents=True, exist_ok=True)
    month_lst = AGE_DICT[lang]     # try out different age ranges
    count = args.count
    # further process the data



    for header in tqdm(header_lst):
        count_test_file = Path(score_dir) / f"{header}.csv"
        count_all_file = Path(args.freq_path) / f"{header}.csv"
        # get adjusted count grouped by month
        count_loader = MonthCounter(
            gen_file=gen_file,
            est_file=est_file,
            test_file=test_file,
            count_all_file=count_all_file,
            count_test_file=count_test_file,
            header=header,
            month_range=month_lst,
            count = count
        )
        count_loader.get_count()  # get the adjusted test count


if __name__ == "__main__":
    args = sys.argv[1:]
    main()
