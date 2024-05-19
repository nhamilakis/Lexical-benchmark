"""match freq based on between human and machine cdi"""
import argparse
import pandas as pd
import sys
import numpy as np
from pathlib import Path
from lm_benchmark.settings import ROOT
from lm_benchmark.analysis.freq.freq_util import bin_stats,loss,init_index,swap_index,get_freq
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--CDI_path",
                        default=f'{ROOT}/datasets/processed/CDI/')
    parser.add_argument("--human_freq",
                        default=f'{ROOT}/datasets/processed/freq/CHILDES_adult.csv')
    parser.add_argument("--machine_freq",
                        default=f'{ROOT}/datasets/processed/freq/3200.csv')
    parser.add_argument("--lang", type=str, default='AE')
    parser.add_argument("--test_type", type=str, default='exp')
    parser.add_argument("--sampling_ratio", type=int, default=1)
    parser.add_argument("--nbins", type=int, default=6)
    return parser.parse_args()


def annotate_freq(CDI_file,human_freq):
    CDI_file = pd.read_csv(CDI_file)
    human_freq = pd.read_csv(human_freq)
    merged_df = pd.merge(CDI_file,human_freq, on='word', how='left')
    cleaned_df = merged_df.dropna()
    return cleaned_df


def match_sample(dataref,datasam,sampling_ratio,nbins,n=100000):
    """"Match a source distribution to a target distribution by sampling randomly from the (larger)
    source distribution in order to minimize a given loss function
    returns the index to the source distribution, the loss and various stats
    The two distributions have the same number of samples (if sampling_ratio larger than one, the returned distribution can contain more samples than the target distribution)
    """
    # convert into freq_m
    dataref = get_freq(dataref,'count')
    datasam = get_freq(datasam,'count')
    lenref=len(dataref)
    lensam=len(datasam)
    assert lenref*sampling_ratio<lensam
    refstat=bin_stats(dataref,nbins)
    pidx,nidx=init_index(lensam,lenref*sampling_ratio)
    data=datasam[pidx]
    datastat=bin_stats(data,nbins)
    lbest=loss(refstat,datastat)
    for i in range(n):
        pidx1,nidx1=swap_index(pidx,nidx)
        data=datasam[pidx1]
        datastat=bin_stats(data,nbins)
        l1=loss(refstat,datastat)
        if lbest>l1:
            lbest=l1
            pidx,nidx=np.array(pidx1,copy=True),np.array(nidx1,copy=True)
    teststat = bin_stats(datasam[pidx], nbins)
    refstat['set'] = 'human'
    teststat['set'] = 'machine'
    stat = pd.concat([refstat,teststat])
    return pidx,lbest,stat

def main():
    """ Run the GoldReference loader and write results to a file """
    args = arguments()
    lang = args.lang
    machine_freq = Path(args.machine_freq)
    CDI_file = Path(args.CDI_path).joinpath(lang + '_' + args.test_type + '_human.csv')
    CDI_stat = Path(args.CDI_path).joinpath(lang + '_' + args.test_type + '_stat.csv')
    human_freq = Path(args.human_freq)
    machine_CDI = Path(args.CDI_path).joinpath(lang + '_' + args.test_type + '_machine.csv')
    # match human-CDI and CHIDLES
    target = annotate_freq(CDI_file, human_freq)
    machine_freq = pd.read_csv(machine_freq)
    # match files
    pidx, lbest, stat = match_sample(target, machine_freq, args.sampling_ratio, args.nbins)
    # save the files
    machine_freq.iloc[pidx].to_csv(machine_CDI)
    stat.to_csv(CDI_stat)





if __name__ == "__main__":
    args = sys.argv[1:]
    main()


