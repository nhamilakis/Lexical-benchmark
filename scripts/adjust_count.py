"""convert word count into accumulated monthly count."""
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from lm_benchmark.analysis.score_util import MonthCounter
from lm_benchmark.settings import ROOT


def arguments() -> argparse.Namespace:
    """Build & Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file", default=f"{ROOT}/datasets/processed/generation/unprompted_Transformer.csv")
    parser.add_argument("--est_file", default=f"{ROOT}/datasets/raw/vocal_month.csv")
    parser.add_argument("--CDI_path", default=f"{ROOT}/datasets/processed/CDI/")
    parser.add_argument("--freq_path", default=f"{ROOT}/datasets/processed/month_count/")
    parser.add_argument("--prompt_type", default="unprompted")
    parser.add_argument("--lang", default="BE")
    parser.add_argument("--set", default="machine")
    parser.add_argument("--header_lst", default=["unprompted_0.3","unprompted_0.6","unprompted_1.0","unprompted_1.5"])
    parser.add_argument("--count", default=False)
    return parser.parse_args()
def main() -> None:
    """Run the GoldReference loader and write results to a file."""
    args = arguments()
    model = args.gen_file.split('/')[-1][:-4].split('_')[1]
    gen_file = Path(args.gen_file)
    est_file = Path(args.est_file)
    lang = args.lang
    header_lst = args.header_lst
    test_file = Path(args.CDI_path) / f"{lang}_exp_{args.set}.csv"
    month_lst = [6, 36]
    count = args.count

    for header in tqdm(header_lst):
        score_dir = Path(f"{args.freq_path}/{args.prompt_type}/{model}")
        count_all_file = Path(score_dir) / f"{header}.csv"
        count_test_dir = Path(score_dir) / lang
        count_test_file = Path(score_dir) / lang / f"{header}.csv"
        score_dir.mkdir(parents=True, exist_ok=True)
        count_test_dir.mkdir(parents=True, exist_ok=True)

        # get adjusted count grouped by month
        count_loader = MonthCounter(
            gen_file=gen_file,
            est_file=est_file,
            test_file=test_file,
            count_all_file=count_all_file,
            count_test_file=count_test_file,
            header=header,
            month_range=month_lst,
            count=count
        )
        count_loader.get_count()  # get the adjusted test count


if __name__ == "__main__":
    args = sys.argv[1:]
    main()
