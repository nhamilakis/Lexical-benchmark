import argparse
import sys
from lm_benchmark.settings import ROOT
from lm_benchmark.datasets.machine_cdi.probe_util import *
def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Select probe set that is unique to the target dir')
    parser.add_argument('--input_dir', type=str, default=f"{ROOT}/datasets/raw/100",
                        help='batch dir')
    parser.add_argument('--output_dir', type=str, default=f"{ROOT}/datasets/processed/CDI/100",
                        help='unique set dir')
    parser.add_argument('--gen_dir', type=str, default=f"{ROOT}/datasets/processed/generation/100",
                        help='unique set dir')
    return parser.parse_args(argv)


def main(argv):
    # load args
    args = parseArgs(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    gen_dir = Path(args.gen_dir)
    filenames = sort_files(input_dir,'txt')
    gen_filenames = sort_files(gen_dir, 'csv')
    # load files
    files = load_files(filenames,input_dir)
    gen_files = load_files(gen_filenames, gen_dir)
    print(f'Loading files from {input_dir}')
    probe_files = select_probe_set(files, output_dir)
    print(f'Saved the selected probing set to {output_dir}')
    result = compare_scores(probe_files,gen_files)




if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)





