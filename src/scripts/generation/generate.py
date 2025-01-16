"""Generate word-like units based on the target distr final results should be model by model."""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lexical_benchmark.gen import generate as lm_generate
from lexical_benchmark.utils import hf_util
from lexical_benchmark.utils.format_util import str_to_bool


def parse_args():
    # Run parameters
    parser = argparse.ArgumentParser(description="Finetune decoder-onlyls models")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/scratch1/projects/lexical-benchmark/v2/models/ChildRealistic/by_month/EN/12/00/LSTM/checkpoint-75000",
        help="Path to the base LM",
    )
    parser.add_argument(
        "--generation_path",
        type=str,
        default="/scratch1/projects/lexical-benchmark/v2/gen/ChildRealistic/by_month/EN/12/00/LSTM",
        help="Path to the generated texts",
    )
    parser.add_argument(
        "--gen_file",
        type=str,
        default="/scratch1/projects/lexical-benchmark/v2/gen/CHILDES_model.csv",
        help="Path to the generated texts",
    )
    parser.add_argument("--temp_lst", type=list, default=[0.3, 0.6, 1.0, 1.5], help="target month model")
    parser.add_argument("--gen_name", type=str, default="gen.csv", help="gen file name")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--AddedTokens", default=["'", "|"], help="A list of added special tokens")
    parser.add_argument("--SAVE_INTERVAL", default=2,  type=int, help="The number of rows to save")
    parser.add_argument("--debug", default="False", help="if debug, generate first 10 sentences")
    return parser.parse_args()


def split_dataframe(df, n_rows):
    """Split a dataframe into a list of subdataframes with approximately n_rows each.

    Args:
        df (pd.DataFrame): Input dataframe
        n_rows (int): Approximate number of rows for each subdataframe

    Returns:
        list: List of subdataframes
    """
    total_rows = len(df)
    n_splits = (total_rows + n_rows - 1) // n_rows

    print(f"Splitting dataframe of {total_rows} rows into {n_splits} parts")

    dfs = []
    for i in range(n_splits):
        start_idx = i * n_rows
        end_idx = min((i + 1) * n_rows, total_rows)
        subdf = df.iloc[start_idx:end_idx].copy()
        dfs.append(subdf)
        print(f"Split {i+1}: {len(subdf)} rows (indices {start_idx} to {end_idx-1})")

    return dfs


def main():
    # Args parser
    args = parse_args()
    device = 0 if torch.cuda.is_available() else "cpu"
    seed = args.seed
    # set the constant random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path = args.model_path
    gen_name = args.gen_name
    model_type = Path(args.generation_path).name
    try:
        month = int(Path(args.generation_path).parents[1].name)
    except ValueError as e:
        raise ValueError(f"Parent folder of {args.generation_path} does not contain month info !") from e

    print(f"{month=}")

    # make directory if not existing
    generation_path = Path(args.generation_path)
    generation_path.mkdir(parents=True, exist_ok=True)
    logger = lm_generate.setup_logging(args.generation_path)
    logger.info(f"Starting generation with arguments: {args}")

    #################
    # load file
    #################
    # intialize the temp list and only update when there is existing col
    temp_lst = args.temp_lst

    print(f"Generating from {args.gen_file}")
    data = pd.read_csv(args.gen_file)
    # filter by month
    df = data[data["model"] == month]

    if str_to_bool(args.debug):
        df = df.head(20)
        gen_name = gen_name.split(".")[0] + "_debug.csv"
        print("Entering debugging mode!")
        print(df)

    # load tokenizer
    tokenizer = hf_util.load_char_tokenizer(model_max_length=2048, special_token_lst=args.AddedTokens)
    print("Tokenizer loaded!")

    # load model
    model = lm_generate.load_model(model_path, model_type, device)
    print("Model loaded!")

    # perform the temperature samplign across the given list
    temp_columns = [f"unprompted_{temp}" for temp in temp_lst]
    # segment the df into different subdf
    gen = pd.DataFrame()
    dfs = split_dataframe(df, args.SAVE_INTERVAL)
    for df in dfs:
        df[temp_columns] = df["sent_len"].apply(lambda x: pd.Series(lm_generate.generate(x, tokenizer, model, device, temp_lst)))
        print(df)
        logger.info(f"Generated texts saved to {generation_path}")
        gen = pd.concat([gen, df])
        gen.to_csv(generation_path / "gen_intermediate.csv")
        print(f"Having saved the generated file to {generation_path}")

    gen.to_csv(generation_path / gen_name)
    print(f"Having saved the generated file to {generation_path}")
    logger.info(f"Generated texts saved to {generation_path}")



if __name__ == "__main__":
    main()
