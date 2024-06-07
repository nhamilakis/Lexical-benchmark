import argparse
import sys
import pandas as pd
from pathlib import Path

from lm_benchmark.settings import ROOT
from lm_benchmark.utils import TokenCount
from lm_benchmark.analysis.freq_util import count_ngrams

def arguments() -> argparse.Namespace:
    """Build & Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", default=f"{ROOT}/datasets/raw/train/3200.csv")
    parser.add_argument("--target_file", default=f"{ROOT}/datasets/processed/freq/3200_3gram.csv")
    parser.add_argument("--header", default="train")
    parser.add_argument("--ngram", default=3)
    return parser.parse_args()


def main() -> None:
    """Run the GoldReference loader and write results to a file."""
    args = arguments()

    target = Path(args.target_file)
    src_file = args.src_file
    header = args.header  # only when loading csv file
    ngram = args.ngram
    print(f'Loading text from {src_file}')

    if ngram == 1:
        if src_file.endswith("txt"):
            count_df = TokenCount.from_text_file_train(src_file)
            #count_df = TokenCount.from_text_file(src_file)
        elif src_file.endswith("csv"):
            count_df = TokenCount.from_df(src_file, header)

    elif ngram > 1:
        if src_file.endswith("txt"):
            header = 0 # load the text file as csv directly
        sentences = pd.read_csv(src_file)[header].tolist()
        count_df = count_ngrams(sentences, ngram)

    else:
        print('The ngram number should be an integer and above 0!')

    count_df.df.to_csv(target, index=False)
    print(f'Writing freq file to {target}')

if __name__ == "__main__":
    args = sys.argv[1:]
    main()


