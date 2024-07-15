"""Construct same-sized datasets in/out of the domain."""

import argparse
import os
import re
import string
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
from tqdm import tqdm

from lm_benchmark import settings


def load_metadata(meta_data_path: Path, text_dir: Path) -> pd.DataFrame:
    """Load metadata for a fine-grained match."""
    meta_data = pd.read_csv(meta_data_path)

    # Extract filenames
    meta_data["filename"] = meta_data["text_path"].apply(lambda x: Path(x).name)

    # All files in text_dir
    print(f"Counting token_num from {text_dir}")
    all_filenames = {f.name for f in text_dir.iterdir()}
    selected_data = meta_data[meta_data["filename"].isin(all_filenames)]

    num_tokens = []
    for filename in selected_data["filename"].tolist():
        frame = txt2csv(text_dir / filename)
        num_tokens.append(frame["num_tokens"].tolist()[0])
    selected_data["num_tokens"] = num_tokens
    selected_data.to_csv(meta_data_path)
    return selected_data


# TODO(@Jing): this clean-up has some logic issues & might remove too much content
def clean_text(loaded: list[str]) -> list[str]:
    """Remove digits and punct of a text string.

    Returns
    -------
    a list of the cleaned strings

    """
    # puctuation remover using translations
    translator = str.maketrans("", "", string.punctuation + string.digits)
    translator[ord("-")] = ord(" ")  # Replace hyphen with blank space

    result = [line for line in loaded if line.strip()]
    cleaned_text = []
    for sent in tqdm(result):
        # Filter out non-ASCII characters
        clean_string = "".join(char for char in sent if ord(char) < 128)
        # Filter punctuation using translator
        clean_string = clean_string.translate(translator).lower()
        # ISSUE: what does this remove ? spaces ? use str.strip for that redundant
        clean_string = re.sub(r"\s+", " ", clean_string)
        # Filter trailing spaces
        clean_string = clean_string.strip()
        # Append to final text
        cleaned_text.append(clean_string)
    return cleaned_text


def count_token(text: str) -> int:
    """Return number of word-tokens in a string."""
    return len(text.split())


def txt2csv(txt_file: Path) -> pd.DataFrame:
    """Load a txt file into csv dataframe.

    Columns: filename;train;num_token
    """
    # read train filename
    cleaned_lines = clean_text(txt_file.read_text(encoding="utf8").split())
    frame = pd.DataFrame(cleaned_lines)
    # assign column headers
    frame = frame.rename(columns={0: "train"})
    frame["num_tokens"] = frame["train"].apply(count_token)
    frame.insert(loc=0, column="filename", value=txt_file.name)

    return frame


def remove_sublist(main_list: t.Sequence[t.Hashable], sublist: t.Sequence[t.Hashable]) -> list[t.Hashable]:
    """Remove a sublist from the main list if it's found."""
    sublist_len = len(sublist)
    m_list = list(main_list)
    for i in range(len(main_list) - sublist_len + 1):
        if main_list[i : i + sublist_len] == sublist:
            return m_list[:i] + m_list[i + sublist_len :]
    return list(main_list)  # Sublisst not found


def cut_df(df: pd.DataFrame, target_cum_sum: pd.DataFrame, header: str = "num_tokens") -> pd.DataFrame:
    """Cut df rows until it has reached the target value."""
    # Calculate cumulative sum
    cum_sum = df[header].cumsum()
    # Find the index where the cumulative sum exceeds or equals the target value
    index_to_cut = cum_sum[cum_sum >= target_cum_sum].index.min()
    # If no index found, keep all rows
    if pd.isna(index_to_cut):
        index_to_cut = len(df)
    # Remove rows after the index_to_cut
    return df.iloc[:index_to_cut]


def match_dataframes(dfa: pd.DataFrame, dfb: pd.DataFrame) -> pd.DataFrame:
    """Match files based on genre."""
    matched_rows = []
    for genre in dfa["genre"].unique():
        dfa_genre = dfa[dfa["genre"] == genre]
        dfb_genre = dfb[dfb["genre"] == genre]

        if len(dfb_genre) < len(dfa_genre):
            raise ValueError(f"Not enough rows in dfB to match genre '{genre}' in dfA")

        cost_matrix = np.abs(dfa_genre["num_tokens"].to_numpy()[:, None] - dfb_genre["num_tokens"].to_numpy())
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_rows.append(dfb_genre.iloc[col_ind])

    # Return concatenation of matched rows
    return pd.concat(matched_rows)


def get_ind_mat(
    filename_path: Path,
    train_freq_dir: Path,
    file: str,
    text_dir: Path,
    meta_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the target pseudo dataset to estimate oov token freq."""
    # read train filename
    file_lst = pd.read_csv(filename_path, header=None)[0].tolist()

    # remove the ones that are already in the train list
    all_file_lst = [f.name for f in text_dir.iterdir()]
    candi_lst = remove_sublist(all_file_lst, file_lst)

    # match the genre
    genre_candi = meta_data[meta_data["filename"].isin(candi_lst)]
    genre_target = meta_data[meta_data["filename"].isin(file_lst)]

    # count token numbers
    matched_df = match_dataframes(genre_target, genre_candi)

    # get the total number of tokens
    train_num = pd.read_csv(train_freq_dir / file)["num_tokens"].sum()

    # get constructed set
    train_frames = [txt2csv(text_dir / file) for file in matched_df["filename"].tolist()]
    train_frame = pd.concat([*train_frames])

    return train_frame, train_num


def get_ood_mat(text_path: Path, train_freq_dir: Path, out_dir: Path) -> None:
    """Construct the target pseudo dataset from CHIDLES transcript."""
    # get constructed set
    frame = pd.read_csv(text_path)
    frame = frame.dropna()
    frame = frame.sample(frac=1, random_state=66).reset_index(drop=True)
    # loop train_freq file
    for file in tqdm(os.listdir(train_freq_dir)):
        # count token numbers
        train_num = pd.read_csv(train_freq_dir / file)["num_tokens"].sum()
        oov_sum = 0
        train_frame = pd.DataFrame()
        n = 0
        while oov_sum < train_num:
            oov_sum += frame["num_tokens"].sum()
            train_frame = pd.concat([train_frame, frame])
            n += 1

        # cut additional line to align with the target train set
        train_frame = cut_df(train_frame, train_num, "num_tokens")
        # print out the utt
        if not out_dir.is_dir():
            out_dir.mkdir(parents=True)
        train_frame.to_csv(out_dir / file)


def parse_args() -> argparse.Namespace:
    """Parse arguments for dataset creation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("train_freq_dir")
    parser.add_argument("out_dir")
    parser.add_argument("input_filename_path")
    parser.add_argument("-m", "--mode", default="ood")
    # TODO(@Jing): What is this ?
    parser.add_argument("-f", "--file", default="400.csv")

    return parser.parse_args()


def main() -> None:
    """Main function allowing to create the dataset."""
    args = parse_args()
    mode = args.mode
    train_freq_dir = Path(args.train_freq_dir)
    out_dir = Path(args.out_dir) / mode

    if mode == "ind":
        meta_data_path = settings.conf.transcript_path
        text_dir = settings.conf.audiobook_txt_path
        if not (meta_data_path / "matched.csv").is_file():
            print("No metadata available, creating...")
            meta_data = load_metadata(meta_data_path / "matched2.csv", text_dir)
        else:
            meta_data = pd.read_csv(meta_data_path / "matched.csv")
        # "/Users/jliu/PycharmProjects/freq_bias_benchmark/data/train/filename/7100.csv"
        filename_path = Path(args.input_filename_path)

        file = args.file
        train_frame, train_num = get_ind_mat(filename_path, train_freq_dir, file, text_dir, meta_data)

        # print out the utt
        if not out_dir.is_dir():
            out_dir.mkdir(parents=True)

        train_frame.to_csv(out_dir / file)
        train_frame = pd.read_csv(out_dir / file)
        train_frame = cut_df(train_frame, train_num)
        train_frame.to_csv(out_dir + file)

    else:
        get_ood_mat(settings.conf.childes_adult_csv_path, train_freq_dir, out_dir)
