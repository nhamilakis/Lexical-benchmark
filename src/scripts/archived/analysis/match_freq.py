#!/usr/bin/env python
"""match freq based on between human and machine cdi."""

import argparse
import warnings

import numpy as np
import pandas as pd
from rich.console import Console

from lexical_benchmark.datasets import wordstats
from lexical_benchmark.settings import CONTENT_POS
from lexical_benchmark.utils import stat_tools

try:
    import polars as pl
except ImportError:
    print("Install polars for dataframe loading !")
    raise

warnings.filterwarnings(action="ignore", category=FutureWarning, message=r".*behavior of DataFrame.sum.*")
console = Console()


def arguments() -> argparse.Namespace:
    """Build & Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="EN")
    parser.add_argument("--test-type", type=str, default="exp", choices=["recep", "exp"])
    parser.add_argument("-s", "--sampling_ratio", type=int, default=1)  # To test 1 & 2
    parser.add_argument("--nbins", type=int, default=6)  # # 6 or 12
    parser.add_argument("-n", "--number-of-iterations", type=int, default=100000)  # needs documentation

    return parser.parse_args()


def match_sample(
    dataref: pd.DataFrame,  # "word", "freq"
    datasam: pd.DataFrame,  # "word", "freq"
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
    dataref = np.log10(dataref["freq"])  #'freq_m'
    datasam = np.log10(datasam["freq"])  #'freq_m'

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


def load_stella_60_00(sampling_ratio: int, lang: str = "EN") -> pl.DataFrame:
    """Load Word-Count data for STELA/by_month/60/00."""
    dataset = wordstats.WordStatsDataset(lang=lang, sampling_ratio=sampling_ratio)
    wf = pl.read_csv(
        dataset.word_frequencies.stela_by_month_60_00,
        has_header=True,
    )
    pos_df = pl.read_csv(
        dataset.pos_view.stela,
        has_header=True,
    )
    word_filter = stat_tools.WordFilter(wf)
    wf = word_filter.filter_words(pos_df=pos_df, content_pos=CONTENT_POS, dataset_name="stela")
    total_count = wf["count"].sum()
    return wf.with_columns(((pl.col("count") / pl.lit(total_count)) * pl.lit(1_000_000)).alias("freq"))


def load_cdi_childes_data(sampling_ratio: int, lang: str = "EN") -> pl.DataFrame:
    """Load the CDI/CHILDES Word-Count-Frequency data."""
    dataset = wordstats.WordStatsDataset(lang=lang, sampling_ratio=sampling_ratio)
    wf_all = pl.read_csv(
        dataset.word_frequencies.childes_adult,
        has_header=True,
    )
    wf = pl.read_csv(
        dataset.word_frequencies.cdi_childes,
        has_header=True,
    )
    word_filter = stat_tools.WordFilter(wf)
    pos_df = pl.read_csv(
        dataset.pos_view.childes_adult,
        has_header=True,
    )
    wf = word_filter.filter_words(pos_df=pos_df, content_pos=CONTENT_POS, dataset_name="child")
    total_count = wf_all["count"].sum()
    # filter freq
    return wf.with_columns(((pl.col("count") / pl.lit(total_count)) * pl.lit(1_000_000)).alias("freq"))


def load_childrealistic_60_00_data(sampling_ratio: int, lang: str = "EN") -> pl.DataFrame:
    """Load the Childrealistic/EN Word-Count data."""
    dataset = wordstats.WordStatsDataset(lang=lang, sampling_ratio=sampling_ratio)
    wf = pl.read_csv(
        dataset.word_frequencies.child_realistic_by_month_60_00,
        has_header=True,
    )
    word_filter = stat_tools.WordFilter(wf)
    pos_df = pl.read_csv(
        dataset.pos_view.child_realistic,
        has_header=True,
    )
    wf = word_filter.filter_words(pos_df=pos_df, content_pos=CONTENT_POS, dataset_name="child")
    total_count = wf["count"].sum()
    return wf.with_columns(((pl.col("count") / pl.lit(total_count)) * pl.lit(1_000_000)).alias("freq"))


def match_sample_wrap(
    dataref: pl.DataFrame,  # "word", "freq"
    datasam: pl.DataFrame,  # "word", "freq"
    sampling_ratio: int,
    nbins: int,
    n: int = 100000,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Wrapper function around match_frequencies."""
    dataref_df = dataref.to_pandas()
    datasam_df = datasam.to_pandas()
    pidx, _, stat = match_sample(
        dataref=dataref_df, datasam=datasam_df, nbins=nbins, sampling_ratio=sampling_ratio, n=n
    )

    datasam_df = datasam_df.iloc[pidx]
    return pl.from_pandas(datasam_df), pl.from_pandas(stat)


def main() -> None:
    """Run the GoldReference loader and write results to a file."""
    args = arguments()

    wordstats_dataset = wordstats.WordStatsDataset(sampling_ratio=args.sampling_ratio)

    ## Load Machine Word-Count & compute frequencies
    machine_freq = load_stella_60_00(sampling_ratio=args.sampling_ratio)

    ## Load HumanRealistic Word-Count & compute frequencies
    human_realistic = load_childrealistic_60_00_data(sampling_ratio=args.sampling_ratio)

    ## Load CDI Word-Count (Childes) & compute frequencies
    cdi_data_childes = load_cdi_childes_data(sampling_ratio=args.sampling_ratio)

    # MATCHING SAMPLES CDI(CHILDES) - Machine
    with console.status("Matching frequencies [CDI - Machine]..."):
        machine_matched, machine_stats = match_sample_wrap(
            dataref=cdi_data_childes,
            datasam=machine_freq,
            nbins=args.nbins,
            sampling_ratio=args.sampling_ratio,
            n=args.number_of_iterations,
        )
    print("Completed Matching frequencies [CDI - Machine]!")

    # MATCHING SAMPLES CDI(CHILDES) - HumanRealistic
    with console.status("Matching frequencies [CDI - HumanRealistic]..."):
        human_realistic_matched, human_realistic_stats = match_sample_wrap(
            dataref=cdi_data_childes,
            datasam=human_realistic,
            nbins=args.nbins,
            sampling_ratio=args.sampling_ratio,
            n=args.number_of_iterations,
        )
    print("Completed Matching frequencies [CDI - HumanRealistic]!")

    # Tag words with their corresponding bin mapped from the frequency bands
    with console.status("Tagging words with corresponding bins..."):
        machine_matched = stat_tools.tag_bins(source=machine_matched, bin_frequencies=machine_stats, set_name="machine")
        human_realistic_matched = stat_tools.tag_bins(
            source=human_realistic_matched, bin_frequencies=human_realistic_stats, set_name="machine"
        )
        cdi_data_childes = stat_tools.tag_bins(
            source=cdi_data_childes, bin_frequencies=human_realistic_stats, set_name="human"
        )

    # Save outputs to disk
    if args.test_type == "exp":
        with console.status(f"Saving files to {wordstats_dataset.matched_root}..."):
            # Save Machine Matched
            wordstats_dataset.matched_frequencies_exp.machine.mk_parent()
            wordstats_dataset.matched_frequencies_exp.machine_stats.mk_parent()

            machine_matched.write_csv(wordstats_dataset.matched_frequencies_exp.machine, include_header=True)
            machine_stats.write_csv(wordstats_dataset.matched_frequencies_exp.machine_stats, include_header=True)

            # Save HumanRealistic Matched
            wordstats_dataset.matched_frequencies_exp.human_reastic_stats.mk_parent()
            wordstats_dataset.matched_frequencies_exp.human_realistc.mk_parent()

            human_realistic_matched.write_csv(
                wordstats_dataset.matched_frequencies_exp.human_realistc, include_header=True
            )
            human_realistic_stats.write_csv(
                wordstats_dataset.matched_frequencies_exp.human_reastic_stats, include_header=True
            )

            # Save new CDI
            cdi_data_childes.write_csv(wordstats_dataset.matched_root / "cdi_childes.csv", include_header=True)

        print(f"Saved files to {wordstats_dataset.matched_root}...")

    else:
        print(f"No target files for {args.test_type}")


if __name__ == "__main__":
    main()
