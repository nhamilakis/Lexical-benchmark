#!/usr/bin/env python
"""match freq based on between human and machine cdi."""

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from lexical_benchmark import settings
from lexical_benchmark.datasets import childes, stella, wordstats
from lexical_benchmark.utils import stat_tools

try:
    import polars as pl
except ImportError:
    print("Install polars for dataframe loading !")
    raise


def arguments() -> argparse.Namespace:
    """Build & Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--CDI_path", default=f"{settings.PATH.dataset_root}/processed/CDI/")
    parser.add_argument("--human_freq", default=f"{settings.PATH.dataset_root}/CHILDES/")
    parser.add_argument("--machine_freq", default=f"{settings.PATH.dataset_root}/processed/freq/3200h.csv")
    parser.add_argument("--lang", type=str, default="BE")
    parser.add_argument("--test-type", type=str, default="exp")
    parser.add_argument("--sampling_ratio", type=int, default=1)
    parser.add_argument("--nbins", type=int, default=6)
    return parser.parse_args()


def annotate_freq(cdi_data: pd.DataFrame, human_freq: pd.DataFrame) -> pd.DataFrame:
    """Annotate Frequencies."""
    merged_df = cdi_data.copy()
    merged_df = merged_df.merge(human_freq, on="word", how="left")
    merged_df.dropna()
    return merged_df


def match_sample(
    dataref: pd.DataFrame,
    datasam: pd.DataFrame,
    sampling_ratio: int,
    nbins: int,
    n: int = 100000,
) -> tuple[int, int, pd.DataFrame]:
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

    refstat = stat_tools.bin_stats(dataref, nbins)
    pidx, nidx = stat_tools.init_index(lensam, lenref * sampling_ratio)
    data = datasam[pidx]
    datastat = stat_tools.bin_stats(data, nbins)
    lbest = stat_tools.loss(refstat, datastat)

    for _ in range(n):
        pidx1, nidx1 = stat_tools.swap_index(pidx, nidx)
        data = datasam[pidx1]
        datastat = stat_tools.bin_stats(data, nbins)
        l1 = stat_tools.loss(refstat, datastat)
        if lbest > l1:
            lbest = l1
            pidx, nidx = np.array(pidx1, copy=True), np.array(nidx1, copy=True)

    teststat = stat_tools.bin_stats(datasam[pidx], nbins)
    refstat["set"] = "human"
    teststat["set"] = "machine"
    stat = pd.concat([refstat, teststat])
    return pidx, lbest, stat


def load_stella_60_00(lang: str = "EN") -> pl.DataFrame:
    """Load Word-Count data for STELA/by_month/60/00."""
    dataset = wordstats.WordStatsDataset(lang=lang)
    return pl.read_csv(
        dataset.word_frequencies.stela_by_month_60_00,
        has_header=True,
    )


def load_cdi_childes_data(lang: str = "EN") -> pd.DataFrame:
    """Load the CDI/CHILDES Word-Count data."""
    dataset = wordstats.WordStatsDataset(lang=lang)
    return pl.read_csv(
        dataset.word_frequencies.cdi_childes,
        has_header=True,
    )

def load_cdi_childrealistic_data(lang: str = "EN") -> pd.DataFrame:
    """Load the CDI/CHILDRealistic Word-Count data."""
    dataset = wordstats.WordStatsDataset(lang=lang)
    return pl.read_csv(
        dataset.word_frequencies.cdi_childrealistic,
        has_header=True,
    )

def load_childrealistic_60_00_data(lang: str = "EN") -> pd.DataFrame:
    """Load the Childrealistic/EN Word-Count data."""
    dataset = wordstats.WordStatsDataset(lang=lang)
    return pl.read_csv(
        dataset.word_frequencies.child_realistic_by_month_60_00,
        has_header=True,
    )


def load_childes_adult_data(lang: str = "EN") -> pd.DataFrame:
    """Load the CHILDES Word-Count data."""
    dataset = wordstats.WordStatsDataset(lang=lang)
    return pl.read_csv(
        dataset.word_frequencies.childes_adult,
        has_header=True,
    )


def main() -> None:
    """Run the GoldReference loader and write results to a file."""
    args = arguments()

    wordstats_dataset = wordstats.WordStatsDataset()

    wordstats_dataset.word_frequencies.stela_by_month_60_00.read_csv()

    ## Load Frequencies & other data
    machine_freq = load_stella_60_00()
    human_freq = load_childes_adult_data()
    # NOTE: are we using this ??
    human_realistic = load_childrealistic_60_00_data()

    # NOTE: these two already have the word frequencies loaded each from a corresponding dataset
    cdi_data_childes = load_cdi_childes_data()
    cdi_data_childrealistic = load_cdi_childrealistic_data()


    # TODO: match files (target here is CDI ?
    # TODO: maybe we should add an option to choose between the two CDI)
    pidx, _, stat = match_sample(target, machine_freq, args.sampling_ratio, args.nbins)

    # TODO: save what ?
    # Is this the target file ?? or does this exist before ?
    machine_cdi_file = Path(args.CDI_path) / f"{args.lang}_{args.test_type}_machine.csv"
    machine_freq.iloc[pidx].to_csv(machine_cdi_file)

    cdi_stat_file = Path(args.CDI_path) / f"{args.lang}_{args.test_type}_stat.csv"
    stat.to_csv(cdi_stat_file)
    #stat.write_csv(cdi_stat_file)
    
if __name__ == "__main__":
    main()
