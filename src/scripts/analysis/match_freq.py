#!/usr/bin/env python
"""match freq based on between human and machine cdi."""

import argparse
import warnings

import numpy as np
import pandas as pd

from lexical_benchmark.datasets import wordstats
from lexical_benchmark.utils import stat_tools

try:
    import polars as pl
except ImportError:
    print("Install polars for dataframe loading !")
    raise

warnings.filterwarnings(action="ignore", category=FutureWarning, message=r".*behavior of DataFrame.sum.*")


def arguments() -> argparse.Namespace:
    """Build & Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="EN")
    parser.add_argument("--test-type", type=str, default="exp", choices=["recep", "exp"])
    parser.add_argument("-s", "--sampling-ratio", type=int, default=1) # To test 1 & 2
    parser.add_argument("--nbins", type=int, default=6) # # 6 or 12
    parser.add_argument("-n", "--number-of-iterations", type=int, default=100000) # needs documentation

    return parser.parse_args()


def annotate_freq(cdi_data: pd.DataFrame, human_freq: pd.DataFrame) -> pd.DataFrame:
    """Annotate Frequencies."""
    merged_df = cdi_data.copy()
    merged_df = merged_df.merge(human_freq, on="word", how="left")
    merged_df.dropna()
    return merged_df


def match_sample(
    dataref: pd.DataFrame, # "word", "freq"
    datasam: pd.DataFrame, # "word", "freq"
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
    dataref = np.log10(dataref["freq"]) #'freq_m'
    datasam = np.log10(datasam["freq"]) #'freq_m'

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

    ## Load Machine Word-Count & compute frequencies
    machine_freq = load_stella_60_00()
    total_count = machine_freq["count"].sum()
    machine_freq = machine_freq.with_columns(
        (pl.col("count") / pl.lit(total_count)).alias("freq")
    )

    ## Load HumanRealistic Word-Count & compute frequencies
    human_realistic = load_childrealistic_60_00_data()
    total_count = human_realistic["count"].sum()
    human_realistic = human_realistic.with_columns(
        (pl.col("count") / pl.lit(total_count)).alias("freq")
    )

    ## Load CDI Word-Count (Childes) & compute frequencies
    cdi_data_childes = load_cdi_childes_data()
    total_count = cdi_data_childes["count"].sum()
    cdi_data_childes = cdi_data_childes.with_columns(
        (pl.col("count") / pl.lit(total_count)).alias("freq")
    )

    # MATCHING SAMPLES CDI - Machine
    pidx, _, stat = match_sample(
        dataref=cdi_data_childes.to_pandas(), datasam=machine_freq.to_pandas(),
        nbins=args.nbins, sampling_ratio=args.sampling_ratio, n=args.number_of_iterations,
    )

    if args.test_type == "exp":
        wordstats_dataset.matched_frequencies_exp.machine.mk_parent()
        machine_freq.to_pandas().iloc[pidx].to_csv(wordstats_dataset.matched_frequencies_exp.machine, index=False)

        wordstats_dataset.matched_frequencies_exp.machine_stats.mk_parent()
        stat.to_csv(wordstats_dataset.matched_frequencies_exp.machine_stats, index=False)
    else:
        print(f"No target files for {args.test_type}")


    # MATCHING SAMPLES CDI - HumanRealistic
    pidx, _, stat = match_sample(
        dataref=cdi_data_childes.to_pandas(), datasam=human_realistic.to_pandas(),
        nbins=args.nbins, sampling_ratio=args.sampling_ratio, n=args.number_of_iterations,
    )
    if args.test_type == "exp":
        wordstats_dataset.matched_frequencies_exp.human_realistc.mk_parent()
        human_realistic.to_pandas().iloc[pidx].to_csv(
            wordstats_dataset.matched_frequencies_exp.human_realistc, index=False
        )

        wordstats_dataset.matched_frequencies_exp.human_reastic_stats.mk_parent()
        stat.to_csv(wordstats_dataset.matched_frequencies_exp.human_reastic_stats, index=False)
    else:
        print(f"No target files for {args.test_type}")


if __name__ == "__main__":
    main()
