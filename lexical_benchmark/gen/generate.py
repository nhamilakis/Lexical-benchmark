"""
generate word-like units based on the target distr
final results should be model by model
"""
import argparse
import random
import logging
import os
import string
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from lexical_benchmark.utils.hf_util import *
from lexical_benchmark.utils.gen_util import *
from transformers import AutoTokenizer, AutoModelForCausalLM


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Finetune decoder-onlyls models')

    parser.add_argument('--model_path', type=str, 
                        default="/scratch1/projects/lexical-benchmark/v2/models/ChildRealistic/by_month/EN/12/00/LSTM/checkpoint-75000",
                        help='Path to the base LM')
    parser.add_argument('--generation_path', type=str, 
                        default="/scratch1/projects/lexical-benchmark/v2/gen/ChildRealistic/by_month/EN/12/00/LSTM",
                        help='Path to the generated texts')
    parser.add_argument('--resume_gen', type=str, 
                        default="",
                        help='Path to the generated texts to be resumed') 
    parser.add_argument('--resume_col', type=list, default=['month','file_id','text','sent_len','model','unprompted_0.3'], 
                        help='column names to preserve') 
    parser.add_argument('--gen_file', type=str, 
                        default="/scratch1/projects/lexical-benchmark/v2/gen/CHILDES_model.csv",
                        help='Path to the generated texts') 
    parser.add_argument('--temp_lst', type=list, default=[0.3,0.6,1.0,1.5],
                        help='target month model')
    parser.add_argument('--gen_name', type=str, default='gen.csv',
                        help='gen file name')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--AddedTokens', default = ['\'','|'],
                      help='A list of added special tokens')
    parser.add_argument('--debug', default="False",
                        help='if debug, generate first 10 sentences')
    return parser.parse_args(argv)



def setup_logging(output_dir: str):
    """Setup logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "inference.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)




def load_model(
    model_path: str,
    model_type: str = 'LSTM',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> PreTrainedModel:
    """Load the trained model."""
    try:
        if model_type.lower() == 'lstm':
            config = LSTMConfig.from_pretrained(model_path)
            model = LSTMForLanguageModeling.from_pretrained(model_path, config=config)
        else:
            # Load default transformer model
            model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")




def generate(word_num, tokenizer, model, device, temp_lst):
    """
    Transformer model generation
    Generate sequences for a given number of `|` tokens for each temperature in `temp_lst`.
    Returns a dictionary with column names as keys and generated sequences as values.
    """
    results = {}
    # only generate from the lower-cased characters
    random_token_id = random.randint(0, 25)
    
    for temp in temp_lst:
        bar_count = 0
        prev_bar = False

        # Decode the initial token 
        input_ids = torch.tensor([[random_token_id]]).to(device)
        gen = tokenizer.decode(random_token_id)

        # Start generating tokens iteratively
        while bar_count < word_num:
            # Generate the next token(s)
            outputs = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + 1, # Increment by 1 token
                num_beams=1,
                num_return_sequences=1,
                temperature=temp,
                top_k=0,
                top_p=1,
                do_sample=True,
                early_stopping=False,
            )
            
            # Get the newly generated token
            new_token = outputs[0, -1].item() # Last token in the generated sequence
            decoded_token = tokenizer.decode(new_token)
            
            # Check if the token is `|` and avoid consecutive `|` tokens
            if decoded_token == '|':
                if not prev_bar:
                    bar_count += 1
                    prev_bar = True # Mark that we've generated a `|`
            else:
                prev_bar = False # Reset if it's not a `|`
                
            # Add the decoded token to the generated sequence
            gen += decoded_token
            # Update input_ids for the next token generation
            input_ids = torch.cat((input_ids, outputs[0, -1:].unsqueeze(0)), dim=1)
            
        # Add the result to the dictionary with the appropriate column name
        column_name = f'unprompted_{temp}'
        results[column_name] = gen

    return results

    

def main(argv):
    # Args parser
    args = parseArgs(argv)
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
    model_type = args.generation_path.split('/')[-1]
    month = args.generation_path.split('/')[-3]
    print(month)
    
    # make directory if not existing 
    generation_path = Path(args.generation_path)
    generation_path.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.generation_path)
    logger.info(f"Starting generation with arguments: {args}")

    #################
    # load file
    #################
    # intialize the temp list and only update when there is existing col
    temp_lst = args.temp_lst

    if len(args.resume_gen)>0:
        print(f'Recovering generation file from {args.resume_gen}')
        df = pd.read_csv(args.resume_gen)
        if len(args.resume_col)>0:
            # select the target columns 
            try:
                df = df[args.resume_col]
                temp_lst = [0.6,1.0,1.5]
                print(f'Recovering from the {str(args.resume_col)}')
            except:
                print(f'Given columns do not exist. Loading {str(data.columns)}')
    else:
        print(f'Generating from {args.gen_file}')
        data = pd.read_csv(args.gen_file)
        # filter by month
        df = data[data['model']==int(month)]

    if str_to_bool(args.debug):
        df = df.head(5)
        gen_name = gen_name.split('.')[0] + '_debug.csv'
        print('Entering debugging mode!')
        print(df)

    # load tokenizer
    tokenizer = load_char_tokenizer(model_max_length=2048,special_token_lst=args.AddedTokens)
    print('Tokenizer loaded!')

    # load model
    model = load_model(args.model_path, model_type, device)
    print('Model loaded!')
    
    # perform the temperature samplign across the given list
    temp_columns = [f'unprompted_{temp}' for temp in temp_lst]
    
    df[temp_columns] = df['sent_len'].apply(lambda x: pd.Series(generate(x, tokenizer, model, device, temp_lst))) 
    print(df)
    
    df.to_csv(generation_path/gen_name)
    print(f'Having saved the generated file to {str(generation_path)}')
    logger.info(f"Generated texts saved to {str(generation_path)}")



if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
