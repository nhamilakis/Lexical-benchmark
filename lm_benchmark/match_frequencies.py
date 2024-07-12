"""Match freq based on between human and machine cdi."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lm_benchmark import settings
from lm_benchmark.analysis import frequency_utils


def arguments() -> argparse.Namespace:
    """Build & Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--CDI_path", default=f"{settings.ROOT}/datasets/processed/CDI/")
    parser.add_argument("--human_freq", default=f"{settings.ROOT}/datasets/processed/freq/CHILDES_adult.csv")
    parser.add_argument("--machine_freq", default=f"{settings.ROOT}/datasets/processed/freq/3200h.csv")
    parser.add_argument("--lang", type=str, default="BE")
    parser.add_argument("--test_type", type=str, default="exp")
    parser.add_argument("--sampling_ratio", type=int, default=1)
    parser.add_argument("--nbins", type=int, default=6)
    return parser.parse_args()


def annotate_freq(cdi_file: Path, human_freq: Path) -> pd.DataFrame:
    """Annotate Frequencies."""
    cdi_data = pd.read_csv(cdi_file)
    human_freq_data = pd.read_csv(human_freq)
    merged_df = cdi_data.merge(human_freq_data, on="word", how="left")
    merged_df.dropna()
    return merged_df


def match_sample(
    dataref: pd.DataFrame,
    datasam: pd.DataFrame,
    sampling_ratio: int,
    nbins: int,
    n: int = 100000,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Match a source distribution to a target distribution.

    This is achieved by sampling randomly from the (larger)
    source distribution in order to minimize a given loss function
    returns the index to the source distribution, the loss and various stats
    The two distributions have the same number of samples (if sampling_ratio larger than one,
    the returned distribution can contain more samples than the target distribution).
    """
    # convert into freq_m
    dataref = np.log10(dataref["freq_m"])
    datasam = np.log10(datasam["freq_m"])

    lenref = len(dataref)
    lensam = len(datasam)
    if not lenref * sampling_ratio < lensam:
        raise ValueError("The sampling rate is too high to create matched sets!")

    refstat = frequency_utils.bin_stats(np.array(dataref), nbins)
    pidx, nidx = frequency_utils.init_index(lensam, lenref * sampling_ratio)
    data = datasam[pidx]
    datastat = frequency_utils.bin_stats(np.array(data), nbins)
    lbest = frequency_utils.loss(np.array(refstat), np.array(datastat))

    for _ in range(n):
        pidx1, nidx1 = frequency_utils.swap_index(pidx, nidx)
        data = datasam[pidx1]
        datastat = frequency_utils.bin_stats(np.array(data), nbins)
        l1 = frequency_utils.loss(np.array(refstat), np.array(datastat))
        if lbest > l1:
            lbest = l1
            pidx, nidx = np.array(pidx1, copy=True), np.array(nidx1, copy=True)

    teststat = frequency_utils.bin_stats(np.array(datasam[pidx]), nbins)
    refstat["set"] = "human"
    teststat["set"] = "machine"
    stat = pd.concat([refstat, teststat])
    return pidx, lbest, stat


def main() -> None:
    """Run the GoldReference loader and write results to a file."""
    args = arguments()
    lang = args.lang

    machine_freq_file = Path(args.machine_freq)
    cdi_file = Path(args.CDI_path) / f"{lang}_{args.test_type}_human.csv"
    cdi_stat_file = Path(args.CDI_path) / f"{lang}_{args.test_type}_stat.csv"
    human_freq_file = Path(args.human_freq)
    machine_cdi_file = Path(args.CDI_path) / f"{lang}_{args.test_type}_machine.csv"

    # match human-CDI and CHILDES
    target = annotate_freq(cdi_file, human_freq_file)
    machine_freq = pd.read_csv(str(machine_freq_file))
    target.to_csv(cdi_file)
    # match files
    pidx, _, stat = match_sample(target, machine_freq, args.sampling_ratio, args.nbins)
    # save the files

    machine_freq.iloc[pidx].to_csv(machine_cdi_file)
    stat.to_csv(cdi_stat_file)


if __name__ == "__main__":
    main()
