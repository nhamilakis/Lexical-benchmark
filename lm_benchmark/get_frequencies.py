import argparse
from pathlib import Path

import pandas as pd

from lm_benchmark import nlp_tools, settings
from lm_benchmark.analysis import frequency_utils


def arguments() -> argparse.Namespace:
    """Build & Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", default=f"{settings.PATH.DATA_DIR / 'datasets/raw/train/3200.csv'}")
    parser.add_argument("--target_file", default=f"{settings.PATH.DATA_DIR / 'datasets/processed/freq/3200_3gram.csv'}")
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
    print(f"Loading text from {src_file}")

    if ngram == 1:
        if src_file.endswith("txt"):
            token_count = nlp_tools.TokenCount.from_text_file(src_file)
        elif src_file.endswith("csv"):
            token_count = nlp_tools.TokenCount.from_df(src_file, header)
        token_count.df.to_csv(target, index=False)
        print(f"Writing freq file to {target}")
    elif ngram > 1:
        if src_file.endswith("txt"):
            header = 0  # load the text file as csv directly
        sentences = pd.read_csv(src_file)[header].tolist()
        count_df = frequency_utils.count_ngrams(sentences, ngram)
        count_df.to_csv(target, index=False)
        print(f"Writing freq file to {target}")
    else:
        print("The ngram number should be an integer and above 0!")


if __name__ == "__main__":
    main()
