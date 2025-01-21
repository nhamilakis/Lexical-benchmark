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
from tqdm import tqdm


def parse_args():
    # Run parameters
    parser = argparse.ArgumentParser(description="Finetune decoder-onlyls models")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/scratch1/projects/lexical-benchmark/v2/models/STELATranscriptions2/by_month/EN/36/00/LSTM",
        help="Path to the base LM",
    )
    parser.add_argument(
        "--generation_path",
        type=str,
        default="/scratch1/projects/lexical-benchmark/v2/gen/merged/STELATranscriptions2/by_month/EN/36/00/LSTM",
        help="Path to the generated texts",
    )
    parser.add_argument(
        "--gen_file",
        type=str,
        default="/scratch1/projects/lexical-benchmark/v2/gen/merged/CHILDES_model.csv",
        help="Path to the generated texts",
    )
    parser.add_argument("--temp_lst", type=list, default=[0.3, 0.6, 1.0, 1.5], help="target month model")
    parser.add_argument("--gen_name", type=str, default="gen_test.csv", help="gen file name")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--AddedTokens", default=["'", "|"], help="A list of added special tokens")
    parser.add_argument("--SAVE_INTERVAL", default=2,  type=int, help="The number of rows to save")
    parser.add_argument('--resume', action='store_true', help="if true, resume from intermediate generation")
    parser.add_argument("--debug", default="False", help="if debug, generate first 10 sentences")
    return parser.parse_args()





def generate(word_num, tokenizer, model, device, temp_lst):
    """Transformer model generation for word_num=1 with strict position control."""
    results = {}
    
    # Get model's position embedding limit - force it to be conservative
    max_length = min(1024, getattr(model.config, 'n_positions', 1024))
    random_token_id = random.randint(0, 25)
    for temp in temp_lst:
        
            # Start with single token
            input_ids = torch.tensor([[random_token_id]], device=device)
            gen = tokenizer.decode(random_token_id)
            bar_count = 0
            
            while bar_count < word_num and input_ids.shape[1] < max_length:
                # Create position IDs explicitly
                curr_length = input_ids.shape[1]
                position_ids = torch.arange(curr_length, device=device).unsqueeze(0)
                
                # Generate with strict controls
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        #max_new_tokens=1,  # Generate only one token at a time
                        max_length=curr_length + 1,
                        do_sample=True,
                        temperature=temp,
                        num_beams=1,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,  # Use EOS as pad token
                        position_ids=position_ids,
                        use_cache=True,
                    )
                
                # Get new token
                new_token = outputs[0, -1].item()
                decoded_token = tokenizer.decode(new_token)
                
                # Check for bar token
                if decoded_token == "|":
                    bar_count += 1
                
                # Update sequence
                gen += decoded_token
                input_ids = outputs
                
                # Reset if sequence gets too long
                if input_ids.shape[1] >= max_length - 2:
                    input_ids = input_ids[:, :1]
                    
                # Clear GPU cache periodically
                if len(gen) % 100 == 0:
                    torch.cuda.empty_cache()
            
            results[f"unprompted_{temp}"] = gen
            
    return results




def safe_generate(x, tokenizer, model, device, temp_lst):
    with torch.no_grad():
        try:
            return pd.Series(generate(x, tokenizer, model, device, temp_lst))
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return pd.Series({f"unprompted_{temp}": "" for temp in temp_lst})

# Usage
def process_dataframe(df, tokenizer, model, device, temp_lst):
    
    temp_columns = [f"unprompted_{temp}" for temp in temp_lst]
    
    try:
        # Process in smaller batches
        chunk_size = 10
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size].copy()
            chunk[temp_columns] = chunk["sent_len"].apply(
                lambda x: safe_generate(x, tokenizer, model, device, temp_lst)
            )
            df.loc[chunk.index, temp_columns] = chunk[temp_columns]
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"DataFrame processing error: {str(e)}")
    
    return df


def split_dataframe(df, n_rows)->list:
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

    return dfs





def main():
    # Args parser
    args = parse_args()

    # Check if target file already exists
    generation_path = Path(args.generation_path)
    target_file = generation_path / args.gen_name
    if target_file.exists():
        print(f"Target file {target_file} already exists. Skipping generation.")
        return

    # device = 0 if torch.cuda.is_available() else "cpu"
    device = "cpu"
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
    data = pd.read_csv(args.gen_file).loc[:, 'month':]
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


    gen = process_dataframe(df, tokenizer, model, device, temp_lst)
    
    print(gen)

    
    # perform the temperature samplign across the given list
    temp_columns = [f"unprompted_{temp}" for temp in temp_lst]
    # segment the df into different subdf
    resume_file = generation_path /"gen_intermediate.csv"
    if args.resume and resume_file.is_file():
        gen = pd.read_csv(resume_file).loc[:, "month":]
        # look for the target dataframe
        df = df.iloc[gen.shape[0]:]
        print(f"Generating file from {generation_path} /gen_intermediate.csv!")
    else:
        print("Generating file from scratch!")
        gen = pd.DataFrame()

    dfs = split_dataframe(df, args.SAVE_INTERVAL)

    for df in tqdm(dfs):
        df[temp_columns] = df["sent_len"].apply(lambda x: pd.Series(lm_generate.generate(x, tokenizer, model, device, temp_lst)))
        print(df.head(5))
        logger.info(f"Generated texts saved to {generation_path}")
        gen = pd.concat([gen, df])
        gen.to_csv(generation_path / "gen_intermediate.csv")
        print(f"Having saved the generated file to {generation_path}")

    gen.to_csv(generation_path /gen_name)
    print(f"Having saved the generated file to {generation_path}")
    logger.info(f"Generated texts saved to {generation_path}")
    


if __name__ == "__main__":
    main()
