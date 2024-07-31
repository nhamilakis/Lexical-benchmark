import argparse
import collections
import re
import string
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_arguments() -> argparse.Namespace:
    """Parse Command Line arguments."""
    parser = argparse.ArgumentParser(description="Select test sets by frequency")

    parser.add_argument(
        "--filename_path",
        type=str,
        default="/Users/jliu/PycharmProjects/freq_bias_benchmark/data/train/filename/",
        help="path to corresponding filenames to each model",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/Users/jliu/PycharmProjects/Machine_CDI/Lexical-benchmark_data/train_phoneme/dataset/",
        help="path to raw dataset",
    )
    parser.add_argument(
        "--mat_path",
        type=str,
        default="/Users/jliu/PycharmProjects/freq_bias_benchmark/data/train/train_mat/",
        help="path to preprocessed files",
    )
    parser.add_argument(
        "--utt_path",
        type=str,
        default="/Users/jliu/PycharmProjects/freq_bias_benchmark/data/train/train_utt/",
        help="path to preprocessed files",
    )

    return parser.parse_args()


# preprocess the data
def remove_blank(s: str) -> str:
    """Remove trailing and leading blank spaces."""
    if not isinstance(s, str):
        raise TypeError("Requires an str")
    return s.strip()


def get_len(x: collections.Sized) -> int:
    """Returns the length of an object."""
    try:
        return len(x)
    except TypeError:
        return 0


def count_words(sentence: str) -> int:
    """Split the sentence into words based on whitespace."""
    words = sentence.split()
    return len(words)


def clean_txt(sent: str) -> str:
    """Clean the input string."""
    # Filter out non-ASCII characters
    sent = "".join(char for char in sent if ord(char) < 128)
    # remove punctuations
    translator = str.maketrans("", "", string.punctuation + string.digits)
    translator[ord("-")] = ord(" ")  # Replace hyphen with blank space
    clean_string = sent.translate(translator).lower()
    clean_string = re.sub(r"\s+", " ", clean_string)
    return clean_string.strip()


def preprocess(raw: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Preprocessing of words.

    Returns
    -------
        the cleaned files

    """
    raw = [line.strip() for line in raw if line.strip()]
    processed_without_all = []
    processed_with_all = []
    sent_all = []
    for sent in tqdm(raw):
        clean_string = clean_txt(sent)
        word_lst = clean_string.split(" ")
        # convert into corresponding format string
        processed_with = ""
        processed_without = ""
        for word in word_lst:
            upper_char = " ".join(word).upper()
            if not word.isspace():
                processed_with += upper_char + " | "
                processed_without += upper_char + " "

        sent_all.append(clean_string)
        processed_without_all.append(processed_without)
        processed_with_all.append(processed_with)
    # convert the final results into
    return sent_all, processed_with_all, processed_without_all


def get_utt_frame(sent: list, file: str, all_frame: pd.DataFrame) -> pd.DataFrame:
    """Save the cleaned utt as a dataframe."""
    utt_frame = pd.DataFrame(sent)
    utt_frame = utt_frame.rename(columns={0: "train"})
    utt_frame["filename"] = file
    return pd.concat([all_frame, utt_frame])


def main() -> None:
    """Main function allowing to run preprocessing via command line."""
    args = parse_arguments()  # Parse command line arguments
    filename_path = Path(args.filename_path)
    dataset_path = Path(args.dataset_path)
    mat_path = Path(args.mat_path)
    utt_path = Path(args.utt_path)

    if not mat_path.is_dir():
        mat_path.mkdir(parents=True)

    if not utt_path.is_dir():
        utt_path.mkdir(parents=True)

    for file in filename_path.iterdir():
        # loop over different hours
        if file.suffix == ".csv":
            # load training data based on filename list
            file_lst = pd.read_csv(file, header=None)
            all_frame = pd.DataFrame()
            train = []
            for txt_file in file_lst[0]:
                with (dataset_path / txt_file).open("r") as f:
                    raw = f.readlines()
                    sent_all, processed_with, _ = preprocess(raw)
                    all_frame = get_utt_frame(sent_all, txt_file, all_frame)
                    train.extend(processed_with)

            # save the utt csv file
            all_frame.to_csv(utt_path / file.name)
            print(f"Finish prepare utt for {file}")

            out_path = mat_path / file.stem
            if not out_path.is_dir():
                out_path.mkdir(parents=True)

            # Open the file in write mode
            with (out_path / "data.txt").open("w") as f:
                # Write each element of the list to the file
                for item in train:
                    f.write(f"{item}\n")
            print(f"Finish preprocessing {file}")


if __name__ == "__main__":
    main()
