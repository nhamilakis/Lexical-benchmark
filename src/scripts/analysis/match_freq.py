#!/usr/bin/env python
"""match freq based on between human and machine cdi."""

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from lexical_benchmark import settings
from lexical_benchmark.datasets import childes, stella
from lexical_benchmark.utils import stat_tools


def arguments() -> argparse.Namespace:
    """Build & Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root")
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




def load_chiles_adult(lang: str = "EN") -> pd.DataFrame:
    """Load Word Frequencies for CHILDES."""
    childes_dataset = childes.CHILDESDataset()
    uk_freq_file = childes_dataset.wf.processed("Eng-UK", "adult")
    na_freq_file = childes_dataset.wf.processed("Eng-NA", "adult")

    uk_freq = pd.read_csv(uk_freq_file, header=0)
    na_freq = pd.read_csv(na_freq_file, header=0)

    combined_df = pd.concat([uk_freq, na_freq])
    return combined_df.groupby("word")["freq"].sum().reset_index()


def load_stella_3200h(lang: str = "EN") -> pd.DataFrame:
    """Load Word Frequencies for STELA/EN/3200h."""
    stella_dataset = stella.STELATranscriptDataset()
    transcriptions = stella_dataset.by_month / f"by_month/{lang}/36/00/" / "transcriptions.txt"
    freqs = collections.Counter(transcriptions.read_tokenized())
    return pd.DataFrame(list(freqs.items()), columns=["word", "freq"])


def load_cdi_data(test_type: str, lang: str = "EN") -> pd.DataFrame:
    """Load the CDI data."""
    cdi_data_file = Path(...) / f"{lang}_{test_type}_human.csv"

    return pd.read_csv(cdi_data_file)


def main() -> None:
    """Run the GoldReference loader and write results to a file."""
    args = arguments()

    ## Load Frequencies & other data
    machine_freq = load_stella_3200h()
    human_freq = load_chiles_adult()
    cdi_data = load_cdi_data(lang=args.lang, test_type=args.test_type)

    # match human-CDI and CHILDES
    # TODO: this is already computed elsewhere
    target = annotate_freq(cdi_data, human_freq)

    # Why are we overwriting the cdi ??? should create a new childes annotated CDI file ???
    # TODO: write this into a temp file to not overwrite source
    target.to_csv(cdi_file)


    # match files
    pidx, _, stat = match_sample(target, machine_freq, args.sampling_ratio, args.nbins)

    # save the files
    # Is this the target file ?? or does this exist before ?
    machine_cdi_file = Path(args.CDI_path) / f"{args.lang}_{args.test_type}_machine.csv"
    machine_freq.iloc[pidx].to_csv(machine_cdi_file)

    cdi_stat_file = Path(args.CDI_path) / f"{args.lang}_{args.test_type}_stat.csv"
    stat.to_csv(cdi_stat_file)


if __name__ == "__main__":
    main()
