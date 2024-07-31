# segment the dataframes
from pathlib import Path

import pandas as pd

from lexical_benchmark import settings

from .create_dataset import count_token


def segment_sentences(file_path: Path) -> list[str]:
    """Segment prompts into 3-grams."""
    segments = []
    with file_path.open("r") as file:
        for line in file:
            words = line.lower().strip().split()
            length = len(words)

            # Process the sentence if it has more than 3 words
            if length > 3:
                i = 0
                while i < length:
                    # If remaining words are less than 3, take them as they are
                    if length - i <= 3:
                        segments.append(" ".join(words[i:]))
                        break
                    segments.append(" ".join(words[i : i + 3]))
                    i += 3
            else:
                # If the sentence is 3 words or less, take it as it is
                segments.append(" ".join(words))
    return segments


def get_prompt_stat(prompt_path: Path, set_type: str, *, count: bool = False) -> pd.DataFrame:
    """Get the gen stat based on prompt len."""
    ind_prompt = pd.read_csv(prompt_path)
    if count:
        ind_prompt["num_tokens"] = ind_prompt["prompt"].apply(count_token)  # count the number of tokens
    utt_lst = [
        [1, 2, 3],
        [
            ind_prompt[ind_prompt["num_tokens"] == 1].shape[0],
            ind_prompt[ind_prompt["num_tokens"] == 2].shape[0],
            ind_prompt[ind_prompt["num_tokens"] >= 3].shape[0],
        ],
        [
            ind_prompt[ind_prompt["num_tokens"] == 1]["num_tokens"].sum(),
            ind_prompt[ind_prompt["num_tokens"] == 2]["num_tokens"].sum(),
            ind_prompt[ind_prompt["num_tokens"] >= 3]["num_tokens"].sum(),
        ],
    ]
    utt_frame = pd.DataFrame(utt_lst).T
    utt_frame.columns = pd.Index(["length", "num_utt", "num_tokens"])
    utt_frame["set"] = set_type
    return utt_frame


def filter_prompts(ind_prompt_path: Path, ood_prompt_path: Path) -> tuple(pd.DataFrame, pd.DataFrame):
    """Filtering of prompts (?)."""
    # load files
    ind_prompt = pd.read_csv(ind_prompt_path)
    ood_prompt = pd.read_csv(ood_prompt_path)

    # filter the prompts
    # BUG(@JING): function missing ?
    ind_prompt["prompt"] = ind_prompt["train"].apply(lambda s: get_n_words(s, 3))
    # get the prompt length
    ind_prompt["prompt_len"] = ind_prompt["prompt"].apply(count_token)
    ood_prompt["prompt_len"] = ood_prompt["prompt"].apply(count_token)
    # List of words to check for
    word_list = pd.read_csv(settings.PATH.cdi_root / "AE_exp_machine.csv")["word"].tolist()
    be_word = pd.read_csv(settings.PATH.cdi_root / "BE_exp_machine.csv")["word"].tolist()
    word_list.extend(be_word)
    # Create a regular expression pattern that matches any of the words
    pattern = "|".join([f"\\b{word}\\b" for word in word_list])
    # concat the DataFrame
    filtered_ind = ind_prompt[~ind_prompt["prompt"].str.contains(pattern, case=False, regex=True)]
    filtered_ood = ood_prompt[~ood_prompt["prompt"].str.contains(pattern, case=False, regex=True)]
    """
    filtered_ind.pop('unprompted_0.3')
    filtered_ind.pop('unprompted_1.0')
    """
    filtered_ind.pop("prompt")
    filtered_ood.pop("prompt")
    filtered_frame = pd.concat([filtered_ind, filtered_ood])
    filtered_3 = filtered_frame[filtered_frame["prompt_len"] == 3]
    return filtered_frame, filtered_3
