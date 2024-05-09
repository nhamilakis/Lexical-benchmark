import argparse
import sys
import pandas as pd
from pathlib import Path
from lm_benchmark.analysis.freq import FreqGenerater
from lm_benchmark.datasets.utils import cha_phrase_cleaning

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/corpus/raw/CHILDES.csv')
    parser.add_argument("--target_file", default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/corpus/freq/CHILDES.csv')
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
        src_file = Path(src_file.split('.')[0] + '.csv')
        df.to_csv(src_file)

    if not src_file.endswith('txt'):
        src_file = Path(src_file)

    freq_loader = FreqGenerater(
        raw_csv=src_file,
        header=header
    )
    freq_loader.freq.to_csv(target, index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    main()



'''
print(str(Path('/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/corpus/raw/CHILDES.csv')).endswith('CHILDES.csv'))

import pandas as pd

df = pd.read_csv('/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/corpus/raw/CHILDES.csv')
df['num_tokens'].sum()
'''
