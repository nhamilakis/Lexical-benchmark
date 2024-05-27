import argparse
import sys
from lm_benchmark.settings import ROOT
from lm_benchmark.datasets.machine_cdi.probe_util import *
def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Select probe set that is unique to the target dir')
    parser.add_argument('--input_dir', type=str, default=f"{ROOT}/datasets/raw/100",
                        help='batch dir')
    parser.add_argument('--CDI_dir', type=str, default=f"{ROOT}/datasets/processed/CDI/100",
                        help='unique probe set dir')
    parser.add_argument('--output_dir', type=str, default=f"{ROOT}/datasets/processed/CF/100.csv",
                        help='dir to save stat file')
    parser.add_argument('--gen_dir', type=str, default=f"{ROOT}/datasets/processed/generation/100",
                        help='generation dir')
    return parser.parse_args(argv)


def main(argv):
    # load args
    args = parseArgs(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    CDI_dir = Path(args.CDI_dir)
    gen_dir = Path(args.gen_dir)
    # rename the generation files
    epoch_dict = rename_files(gen_dir)
    result_all = pd.DataFrame()
    probe_files_all = {}
    gen_files_all = {}
    # loop over different epochs; use generations to locate the batch files
    for epoch,filenames in epoch_dict.items():
        # load files
        print(f'Loading reference files from {input_dir}')
        files = load_files(filenames,input_dir,'txt')
        print(f'Loading generation files from {gen_dir}')
        gen_files = load_files(filenames, gen_dir,'csv')
        gen_files_all.update(gen_files)
        # build probe tests
        probe_files = select_probe_set(files, CDI_dir)
        probe_files_all.update(probe_files)
        print(f'Saved the selected probing set to {CDI_dir}')

        # run stat analysis
        result = compare_scores(probe_files,gen_files)
        result['epoch'] = epoch
        result_all = pd.concat([result_all,result])

    # also save the result to the whole
    result = compare_scores(probe_files_all, gen_files_all)
    result['epoch'] = 'all'
    result_all = pd.concat([result_all, result])
    result_all.to_csv(output_dir)
    print('Finished stat analysis')



if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)





