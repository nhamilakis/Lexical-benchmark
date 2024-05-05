import argparse
import sys
import pandas as pd
from pathlib import Path
from lm_benchmark.analysis.freq import FreqGenerater
from lm_benchmark.datasets.utils import cha_phrase_cleaning

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/corpus/raw/3200.txt')
    parser.add_argument("--target_file", default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/corpus/freq/3200.csv')
    parser.add_argument("--header",default='content')
    return parser.parse_args()


def main():
    """ Run the GoldReference loader and write results to a file """
    args = arguments()

    target = Path(args.target_file)
    src_file = args.src_file
    header = args.header
    # clean .txt file
    if src_file.endswith('txt'):
        df = pd.read_csv(src_file, header=None, names=[header])
        df = df[~pd.to_numeric(df[header], errors='coerce').notnull()]    # remove float
        df[header] = df[header].apply(cha_phrase_cleaning)
        src = Path(src_file.split('.')[0] + '.csv')
        df.to_csv(src)

    freq_loader = FreqGenerater(
        raw_csv=src,
        header=header
    )
    freq_loader.gold.to_csv(target, index=False)

if __name__ == "__main__":
    args = sys.argv[1:]
    main()

