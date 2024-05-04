import argparse
import dataclasses
import sys
from pathlib import Path

from lm_benchmark.datasets.human_cdi import POSTypes, AGE_MIN, AGE_MAX, GoldReferenceCSV


@dataclasses.dataclass
class BuildHumanCDI:
    filter_by_word_type: POSTypes = POSTypes.content


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter-by-word-type',
                        choices=[str(s) for s in POSTypes], type=str, default=str(POSTypes.content.value),
                        help='Filter words by word_type')
    parser.add_argument("--min-age", type=int, default=AGE_MIN)
    parser.add_argument("--max-age", type=int, default=AGE_MAX)
    parser.add_argument("--src_file", default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/corpus/AE_exp.csv')
    parser.add_argument("--target_file", default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/human/AE.csv')
    return parser.parse_args()


def main():
    """ Run the GoldReference loader and write results to a file """
    args = arguments()

    src = Path(args.src_file)
    target = Path(args.target_file)
    gd_loader = GoldReferenceCSV(
        raw_csv=src,
        age_min=args.min_age,
        age_max=args.max_age,
        pos_filter_type=POSTypes(args.filter_by_word_type)
    )
    gd_loader.gold.to_csv(target, index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    main()

