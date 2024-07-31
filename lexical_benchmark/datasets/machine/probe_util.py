"""Util functions to create probe sets."""

import collections
import re
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel  # type: ignore[import-untyped]
from tqdm import tqdm

from lexical_benchmark import nlp_tools, plot_util


def rename_files(directory: Path) -> dict[str, list[str]]:
    """Given a directory of gen files, rename the batch number and return a dictionary of filenames by epochs."""
    # List to hold tuples of (base name, number, file path)
    file_info = []
    # Regex to extract the base name and number
    pattern = re.compile(r"(.+?)_(\d+)\.csv")
    # Get all files with the format baseName_number.csv
    for file in directory.rglob("*.csv"):
        match = pattern.match(file.name)
        if match:
            base_name, number = match.groups()
            number = int(number)
            file_info.append((base_name, number, file))

    # Sort the list by base name and then by number
    file_info.sort(key=lambda x: (x[0], x[1]))
    # Dictionary to store filenames by epochs
    filenames_by_epochs = collections.defaultdict(list)
    for base_name, _, file in file_info:
        filenames_by_epochs[base_name].append(file.stem)
        new_index = len(filenames_by_epochs[base_name]) - 1
        new_filepath = directory / f"{base_name}_{new_index}.csv"
        print(f"Renaming {file} to {new_filepath}")
        file.rename(new_filepath)
    return dict(filenames_by_epochs)


def load_files(file_names: list[str], parent_path: Path, suffix: str) -> dict[str, nlp_tools.TokenCount]:
    """Load files as TC objects.

    Arguments:
    ---------
    file_names (list):
        List of filenames in the desired order.
    parent_path (str):
        Path to the common parent directory.
    suffix (str):
        the file suffix.

    Returns:
    -------
        dataframes (list): List of TokwnCounts read from the files.

    """
    dataframes = {}

    for file_name in file_names:
        file_path = parent_path / f"{file_name}.{suffix}"
        if file_path.suffix == ".txt":
            # load as TC object
            word_count = nlp_tools.TokenCount.from_text_file(file_path)
        elif file_path.suffix == ".csv":
            word_count = nlp_tools.TokenCount.from_csv(file_path, "LSTM_segmented")

        word_count.name = file_name
        dataframes[file_name] = word_count

    return dataframes


def select_probe_set(files: dict, out_dir: Path, prop: float) -> dict[str, nlp_tools.TokenCount]:
    """Select probe set.

    returns: a dictionary of unique probe files
    """
    # Create out_dir if it does not exist
    out_dir.mkdir(parents=True, exist_ok=True)
    # Iterate over batches (T-1, T, T+1) recursively
    stat_lst = []
    dataframes = {}
    file_keys = list(files.keys())
    for i in tqdm(range(1, len(file_keys) - 1)):
        # Load word count CSV files for batches T-1, T, T+1
        prev_file = file_keys[i - 1]
        curr_file = file_keys[i]
        next_file = file_keys[i + 1]
        df_prev = files[prev_file].df
        df_curr = files[curr_file].df
        df_next = files[next_file].df

        # Get the words from the current batch
        selected_words = df_curr["word"].tolist()
        # Remove words overlapping with batches T-1 and T+1
        selected_words = [
            word
            for word in selected_words
            if word not in df_prev["word"].tolist() and word not in df_next["word"].tolist()
        ]
        selected_df = df_curr[df_curr["word"].isin(selected_words)]

        # Determine the threshold for the lowest 1-prop count values
        selected_df["count"] = selected_df["count"].astype(float)
        selected_df = selected_df.sort_values(by="count")
        count_threshold = selected_df["count"].quantile(1 - prop)
        # Remove words with counts in the lowest 1-prop
        selected_df = selected_df[selected_df["count"] > count_threshold]

        # Get statistics for the CSV file
        stat_lst.append([curr_file, selected_df.shape[0]])
        # Convert it to TokenCount object
        word_count = nlp_tools.TokenCount()
        word_count.name = curr_file
        word_count.df = selected_df
        dataframes[curr_file] = word_count
        # Write probe set to file
        selected_df.to_csv(out_dir / f"batch_{i}.csv", index=False)
        print(f"Probe set for batch {curr_file} saved to {out_dir}")

    stat_df = pd.DataFrame(stat_lst, columns=["filename", "token_count"])
    filename = f"stat_probe_{prop}.csv"
    stat_df.to_csv(out_dir / filename, index=False)
    return dataframes


# Assuming `stat` is the DataFrame with the required columns
def add_difference(stat: pd.DataFrame) -> pd.DataFrame:
    """Add the difference (??)."""
    # Define the new column names
    new_columns = {
        "freq_score": ["freq_score_learn", "freq_score_forget"],
        "pmiss": ["pmiss_learn", "pmiss_forget"],
        "poov": ["poov_learn", "poov_forget"],
        "pnword": ["pnword_learn", "pnword_forget"],
    }

    # Calculate the learning and forgetting columns
    for score, new_cols in new_columns.items():
        learn_col, forget_col = new_cols
        stat[learn_col] = stat[f"{score}_cur"] - stat[f"{score}_prev"]
        stat[forget_col] = stat[f"{score}_cur"] - stat[f"{score}_next"]
    return stat


def compare_scores(probe_files: dict, gen_files: dict, *, run_stat: bool = False) -> pd.DataFrame:
    """Score comparison."""
    stat = pd.DataFrame(
        columns=[
            "freq_score_prev",
            "pmiss_prev",
            "poov_prev",
            "pnword_prev",
            "freq_score_cur",
            "pmiss_cur",
            "poov_cur",
            "pnword_cur",
            "freq_score_next",
            "pmiss_next",
            "poov_next",
            "pnword_next",
        ]
    )

    for filename, probe_tc in probe_files.items():
        # Select from the corresponding results
        prev_filename = filename.split("_")[0] + "_" + str(int(filename.split("_")[1]) - 1)
        next_filename = filename.split("_")[0] + "_" + str(int(filename.split("_")[1]) + 1)

        msc_prev, osc_prev, nsc_prev = plot_util.tc_compute_miss_oov_rates(
            probe_tc, gen_files[prev_filename], groupbin=1
        )
        msc_cur, osc_cur, nsc_cur = plot_util.tc_compute_miss_oov_rates(probe_tc, gen_files[filename], groupbin=1)
        msc_next, osc_next, nsc_next = plot_util.tc_compute_miss_oov_rates(
            probe_tc, gen_files[next_filename], groupbin=1
        )

        stat_temp = pd.DataFrame(
            [
                [
                    msc_prev["dfreq_score"].tolist()[0],
                    msc_prev["pmiss"].tolist()[0],
                    osc_prev["poov"].tolist()[0],
                    nsc_prev["pnword"].tolist()[0],
                    msc_cur["dfreq_score"].tolist()[0],
                    msc_cur["pmiss"].tolist()[0],
                    osc_cur["poov"].tolist()[0],
                    nsc_cur["pnword"].tolist()[0],
                    msc_next["dfreq_score"].tolist()[0],
                    msc_next["pmiss"].tolist()[0],
                    osc_next["poov"].tolist()[0],
                    nsc_next["pnword"].tolist()[0],
                ]
            ],
            columns=[
                "freq_score_prev",
                "pmiss_prev",
                "poov_prev",
                "pnword_prev",
                "freq_score_cur",
                "pmiss_cur",
                "poov_cur",
                "pnword_cur",
                "freq_score_next",
                "pmiss_next",
                "poov_next",
                "pnword_next",
            ],
        )

        stat = pd.concat([stat, stat_temp], ignore_index=True)

    if not run_stat:
        # add additional difference score
        return add_difference(stat)

    # Run stat analysis
    # BUG(@Jing): DataFrames should not be used as dictionairies, appending is not a good use-case, not optimal
    result = pd.DataFrame(
        columns=[
            "freq_score_learn_p",
            "freq_score_learn_t",
            "freq_score_forget_p",
            "freq_score_forget_t",
            "pmiss_learn_p",
            "pmiss_learn_t",
            "pmiss_forget_p",
            "pmiss_forget_t",
            "poov_learn_p",
            "poov_learn_t",
            "poov_forget_p",
            "poov_forget_t",
            "pnword_learn_p",
            "pnword_learn_t",
            "pnword_forget_p",
            "pnword_forget_t",
        ]
    )

    score_lst = ["freq_score", "pmiss", "poov", "pnword"]
    for score in score_lst:
        t_learn, p_learn = ttest_rel(stat[f"{score}_cur"], stat[f"{score}_prev"])
        t_forget, p_forget = ttest_rel(stat[f"{score}_cur"], stat[f"{score}_next"])

        result.loc[0, f"{score}_learn_p"] = p_learn
        result.loc[0, f"{score}_learn_t"] = t_learn
        result.loc[0, f"{score}_forget_p"] = p_forget
        result.loc[0, f"{score}_forget_t"] = t_forget

    return result
