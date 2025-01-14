import argparse
from pathlib import Path
import subprocess
import sys
from lexical_benchmark.utils.gen_util import str_to_bool
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'


#TODO: check whether this can be directly run

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Finetune decoder-onlyls models')

    parser.add_argument('--root_dir', type=str, 
                        default="/scratch1/projects/lexical-benchmark/v2",
                        help='Path to the base LM')
    parser.add_argument('--dataset', type=str, 
                        default="ChildRealistic",
                        help='Path to the generated texts')
    parser.add_argument('--gen_file', type=str, 
                        default="gen.csv",
                        help='gen file name') 
    parser.add_argument('--lang', type=str, default='EN',
                        help='target language')
    parser.add_argument('--debug', default="False",
                        help='if debug, generate first 10 sentences')
    return parser.parse_args(argv)



def run_command(command):
    subprocess.call(command, shell=True)


def check_model_files(model_path:Path):
    """Check if all necessary files for inference exist."""
    required_files = ['config.json', 'pytorch_model.bin', 'training_args.bin']
    return all((model_path/file).exists() for file in required_files)


def main(argv):
    # Args parser
    args = parseArgs(argv)

    dataset= args.dataset     # ChildRealistic or STELATranscriptions2
    lang = args.lang
    root_dir = Path(args.root_dir)
    model_root = root_dir/'models'/dataset/'by_month'/lang
    gen_root = root_dir/'gen'/dataset/'by_month'/lang
    old_gen = root_dir/'gen'/'backup' / dataset/'by_month'/lang
    model_lst = ['LSTM','trans']

    if str_to_bool(args.debug):
        gen_file = args.gen_file.split('.')[0] + '_debug.csv'
        print('Entering debugging mode!')
    else:
        gen_file = args.gen_file
        print('No debugging!')
    month_lst = [36,30,24]
    # loop over the model directory
    for month in model_root.iterdir():
        if int(month.name) in month_lst:
            print(month.name)
            for chunk in month.iterdir():
                for model in chunk.iterdir():
                    # only select the target model in the given list
                    if model.name in model_lst:
                        # check whether the file has already been generated
                        gen_path = gen_root/month.name/chunk.name/model.name
                        old_gen_path = old_gen/month.name/chunk.name/model.name
                        if (gen_path/gen_file).exists():
                            print(f'There already exists {str(gen_path/gen_file)}. Skipping generation')

                        else:
                            # check whether the model has been trained
                            if check_model_files(model):
                                if (old_gen_path/gen_file).exists():
                                    # resume from the old gen
                                    gen_command = f'python generate.py --model_path {model} --generation_path {gen_path} --resume_gen {old_gen_path/gen_file} --debug {args.debug}'

                                else:
                                    print(f'No {str(old_gen_path/gen_file)}. Generating from scratch!')
                                    # if not, gen from scratch
                                    gen_command = f'python generate.py --model_path {model} --generation_path {gen_path} --debug {args.debug}'
                            
                                run_command(gen_command)

                    

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
