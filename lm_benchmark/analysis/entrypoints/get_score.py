"""
convert word count into accumulated monthly count
"""
import argparse
import sys
from pathlib import Path
from lm_benchmark.analysis.score import MonthCounter


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file",
                        default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/score/generation.csv')
    parser.add_argument("--est_file",
                        default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/score/vocal_month.csv')
    parser.add_argument("--out_path",
                        default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/score/adjusted_count/')
    parser.add_argument("--header",default='content')
    parser.add_argument("--threshold", default=6)
    return parser.parse_args()


def main():
    """ Run the GoldReference loader and write results to a file """
    args = arguments()
    gen_file = Path(args.gen_file)
    est_file = Path(args.est_file)
    header = args.header

    # get freq grouped by month
    count_loader = MonthCounter(
        gen_file=gen_file,
        est_file=est_file,
        header=header,
        threshold=args.threshold
    )

    adjusted_df = count_loader.count_by_month()
    adjusted_df.to_csv(args.out_path + header + '.csv', index=False)
    # select from the eval test



if __name__ == "__main__":
    args = sys.argv[1:]
    main()



