"""Compute core metrics from the generation file """


import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from lexical_benchmark import settings
from lexical_benchmark.utils.analysis_util import split_df_col
from lexical_benchmark.datasets.utils.text_cleaning import char2word



def parse_args():
    # Run parameters
    parser = argparse.ArgumentParser(description="Concartenate generated sequences")

    parser.add_argument(
        "--gen_dir",
        type=str,
        default="gen/merged",
        help="realtive path to the root dir",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="gen/merged",
        help="realtive path to save the concatenated generation",
    )

    parser.add_argument("--filename", type=str, default="gen.csv", help="gen file name")
    parser.add_argument("--lang", type=str, default='EN', help="random seed")
    return parser.parse_args()


def load_dataset():
    """load the dataset into word list"""

    return word_list



def main():
    # Args parser
    args = parse_args()
    gen_dir: Path = settings.PATH.DATA_DIR / args.gen_dir
    gen_dir: Path = settings.PATH.DATA_DIR / args.gen_dir

    gen_all = pd.DataFrame()
    
    # loop over differnt model, chunk
    
    gen_all.to_csv(out_dir/args.filename)                         
    print(f'Saving the concatenated generation to {out_dir/args.filename}')



if __name__ == "__main__":
    main()
