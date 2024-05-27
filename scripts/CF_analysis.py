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
                        help='generation dir')
    return parser.parse_args(argv)


def main(argv):     #TODO: segment different epochs; adapt to the cluster
    # load args
    args = parseArgs(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    gen_dir = Path(args.gen_dir)

    # loop over different epochs; use generations to locate the batch files

    # rename the generation files
    filenames = rename_files(gen_dir)
    # load files
    print(f'Loading reference files from {input_dir}')
    files = load_files(filenames,input_dir,'txt')
    print(f'Loading generation files from {gen_dir}')
    gen_files = load_files(filenames, gen_dir,'csv')
    probe_files = select_probe_set(files, output_dir)
    print(f'Saved the selected probing set to {output_dir}')
    result = compare_scores(probe_files,gen_files)
    print(result)
    print('Finished stat analysis')



if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)





