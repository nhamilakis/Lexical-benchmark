import argparse
import dataclasses
import sys
from pathlib import Path

from lm_benchmark.settings import ROOT, AGE_DICT
from lm_benchmark.datasets.human_cdi import POSTypes, GoldReferenceCSV

@dataclasses.dataclass
class BuildHumanCDI:
    filter_by_word_type: POSTypes = POSTypes.content


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter-by-word-type',
                        choices=[str(s) for s in POSTypes], type=str, default=str(POSTypes.content.value),
                        help='Filter words by word_type')
    parser.add_argument("--lang", type=str, default='AE')
    parser.add_argument("--test_type",type=str, default='exp')
    parser.add_argument("--src_file",type=str, default= f'{ROOT}/datasets/raw/')
    parser.add_argument("--target_file",type=str, default=f'{ROOT}/datasets/processed/CDI/')
    return parser.parse_args()


def main():
    """ Run the GoldReference loader and write results to a file """
    args = arguments()

    src = Path(args.src_file).joinpath(args.lang + '_' + args.test_type + '.csv')
    target = Path(args.target_file).joinpath(args.lang + '_' + args.test_type + '_human.csv')
    print(f'{args.lang} {args.test_type} loaded')
    age_min = AGE_DICT[args.lang][0]
    age_max = AGE_DICT[args.lang][1]

    gd_loader = GoldReferenceCSV(
        raw_csv=src,
        age_min=age_min,
        age_max=age_max,
        pos_filter_type=POSTypes(args.filter_by_word_type)
    )
    gd_loader.gold.to_csv(target, index=False)
    print('Finished preprocessing!')

if __name__ == "__main__":
    args = sys.argv[1:]
    main()