"""
convert word count into accumulated monthly count
"""
import argparse
import sys
from pathlib import Path
from lm_benchmark.analysis.score import MonthCounter

ROOT = '/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp'

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file",
                        default=f'{ROOT}/score/generation.csv')
    parser.add_argument("--est_file",
                        default=f'{ROOT}/score/vocal_month.csv')
    parser.add_argument("--test_file",
                        default=f'{ROOT}/test/test_set/BE_human.csv')
    parser.add_argument("--count_all_file",
                        default=f'{ROOT}/score/count_all/content.csv')
    parser.add_argument("--count_test_file",
                        default=f'{ROOT}/score/count_test/BE_human.csv')
    parser.add_argument("--header",default='content')
    parser.add_argument("--threshold", default=6)
    return parser.parse_args()


def main():
    """ Run the GoldReference loader and write results to a file """
    args = arguments()
    gen_file = Path(args.gen_file)
    est_file = Path(args.est_file)
    test_file = Path(args.test_file)
    count_all_file = Path(args.count_all_file)
    count_test_file = Path(args.count_test_file)
    header = args.header

    # get freq grouped by month
    count_loader = MonthCounter(
        gen_file=gen_file,
        est_file=est_file,
        test_file=test_file,
        count_all_file=count_all_file,
        header=header,
        threshold=args.threshold
    )

    count_test = count_loader.get_count()
    count_test.to_csv(count_test_file)




if __name__ == "__main__":
    args = sys.argv[1:]
    main()
