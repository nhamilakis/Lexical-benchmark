import typing as t
from pathlib import Path

import datasets
import numpy as np
from torch.utils.data import BatchSampler
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.data import DataCollatorForLanguageModeling


class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """Custom Data Collator that randomly joins utterances together to form longer sequences."""

    def __init__(self, tokenizer: AutoTokenizer, max_seq_length: int = 512, **kwargs) -> None:
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.max_seq_length = max_seq_length

    def __call__(self, examples: list[str], *, return_tensors: bool | None = None) -> None:
        """Call method."""
        new_examples = []
        keys = list(examples[0].keys())
        long_examples = {}
        for key in keys:
            long_examples[key] = []
        for example in examples:
            for key in keys:
                long_examples[key].extend(example[key])
        for i in range(0, len(long_examples[keys[0]]), self.max_seq_length):
            new_example = {}
            for key in keys:
                new_example[key] = long_examples[key][i : i + self.max_seq_length]
            new_examples.append(new_example)
        return super().__call__(new_examples, return_tensors=return_tensors)


class CustomBatchSampler(BatchSampler):
    """Custom batch sampler.

    It ensures we get enough data to fill a batch once the collator has joined utterances
    together to sequence of length max_seq_length.
    """

    def __init__(self, sampler, batch_size, drop_last, max_seq_length) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.max_seq_len = max_seq_length
        self.total_batch_size = batch_size * max_seq_length
        # Compute lengths once and store them
        self.lengths = {idx: len(self.sampler.data_source[idx]["input_ids"]) for idx in self.sampler}
        self.total_length = sum(self.lengths.values())

    def __iter__(self) -> t.Iterator[list[str]]:
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                batch = []
                total_len = 0
                try:
                    while total_len < self.total_batch_size:
                        idx = next(sampler_iter)
                        batch.append(idx)
                        length = self.lengths[idx]
                        total_len += length
                    yield batch[:-1]
                    batch = [idx]
                    total_len = length
                except StopIteration:
                    break
        else:
            batch = []
            idx_in_batch = 0
            total_len = 0
            for idx in self.sampler:
                batch.append(idx)
                length = self.lengths[idx]
                total_len += length
                idx_in_batch += 1
                if total_len >= self.total_batch_size:
                    yield batch[:-1]
                    idx_in_batch = 1
                    total_len = length
                    batch = [idx]
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
        if self.drop_last:
            return self.total_length // self.total_batch_size
        return (self.total_length + self.total_batch_size - 1) // self.total_batch_size


class DataPreprocessor:
    """Used to preprocess data."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        get_word_boundaries: bool = True,
    ) -> None:
        # data processing params
        self.max_input_length = 2048
        self.join_utts = None

        self.tokenizer = tokenizer
        self.utterance_boundary_token = tokenizer.eos_token
        self.get_word_boundaries = get_word_boundaries
        if self.get_word_boundaries:
            if "WORD_BOUNDARY" in tokenizer.get_added_vocab():
                self.word_boundary_token = tokenizer.convert_tokens_to_ids("WORD_BOUNDARY")
            elif "W" in tokenizer.get_added_vocab():
                self.word_boundary_token = tokenizer.convert_tokens_to_ids("W")
            else:
                raise ValueError(
                    "Tokenizer does not contain the word boundary token (should be 'W' or"
                    "'WORD_BOUNDARY' in added tokens). Cannot extract or remove word boundaries."
                )

    def __call__(self, examples: list[str]) -> dict[str, t.Any]:
        """The tokenizer should have been configured to add an utterance boundary to the start of each utterance.

        There are three options for joining utterances. If join_utts is None, each example contains a single utterance
        with utterance boundaries at the start and end and padding to the max_input_length:
        e.g. [UTT_BOUNDARY, token1, token2, ..., tokenN, UTT_BOUNDARY, PAD, ..., PAD]

        If join_utts is 'static', all utterances are concatenated and split into chunks of max_input_length:
        e.g. [UTT_BOUNDARY, token1, token2, ..., tokenN, UTT_BOUNDARY, token1, token2, ..., tokenN, UTT_BOUNDARY, ...]
        In this case, only the final few tokens of each chunk will be padded.

        If join_utts is 'dynamic', utterances are concatenated randomly by the DataCollator so that the model always
        sees new combinations of utterances and doesn't overfit to the ordering presented in the dataset. We therefore
        do not need to do anything to the utterances here besides tokenize them.
        """
        if self.join_utts == "static":
            batch = {}
            joined = f" {self.utterance_boundary_token} ".join([utt.strip() for utt in examples["text"]])
            joined = self.tokenizer(joined, truncation=False, padding=False)
            input_ids = joined["input_ids"]
            attention_mask = joined["attention_mask"]

            if self.get_word_boundaries:
                # Create an array of positions that mark the start of a word
                word_start_positions = np.minimum(
                    len(input_ids) - 1, np.where(np.array(input_ids) == self.word_boundary_token)[0] + 1
                )
                word_starts = np.zeros(len(input_ids), dtype=bool)
                word_starts[word_start_positions] = True
                # Every position after a word boundary is also a word start
                word_starts = np.logical_or(word_starts, np.array([0] + input_ids[:-1]) == self.word_boundary_token)
                # Every position after an utterance boundary is a word start
                word_starts = np.logical_or(word_starts, np.array([0] + input_ids[:-1]) == self.tokenizer.eos_token_id)
                # Utterance boundaries are not word boundaries
                word_starts = np.logical_and(word_starts, np.array(input_ids) != self.tokenizer.eos_token_id)

            # Split the long vector into inputs of length max_input_length
            batch["input_ids"] = [
                input_ids[i : i + self.max_input_length] for i in range(0, len(input_ids), self.max_input_length)
            ]
            batch["attention_mask"] = [
                attention_mask[i : i + self.max_input_length] for i in range(0, len(input_ids), self.max_input_length)
            ]
            if self.get_word_boundaries:
                batch["word_starts"] = [
                    word_starts[i : i + self.max_input_length] for i in range(0, len(input_ids), self.max_input_length)
                ]

            return batch

        # If join_utts is None, we add a utterance boundary token to the end of each utterance
        # If join_utts is 'dynamic', we do not need to do anything to the utterances here
        if self.join_utts is None:
            examples["text"] = [
                (examples["text"][i] + " " + self.utterance_boundary_token) for i in range(len(examples["text"]))
            ]

        # If join_utts is 'none' or 'dynamic' we tokenize the utterances individually and return them
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_input_length,
            padding=False,
        )

        if self.get_word_boundaries:
            word_starts_list = []
            for input_ids in tokenized["input_ids"]:
                # Create an array of positions that mark the start of a word
                line_length = len(input_ids)
                word_start_positions = np.where(np.array(input_ids) == self.word_boundary_token)[0] + 1
                word_start_positions = np.minimum(line_length - 1, word_start_positions)

                word_starts = np.zeros(len(input_ids), dtype=np.int8)
                word_starts[word_start_positions] = 1
                # Every position after a word boundary is also a word start
                word_starts = np.logical_or(word_starts, np.array([0] + input_ids[:-1]) == self.word_boundary_token)
                # Every position after an utterance boundary is a word start
                word_starts = np.logical_or(word_starts, np.array([0] + input_ids[:-1]) == self.tokenizer.eos_token_id)
                word_starts = np.logical_and(
                    word_starts, np.array(input_ids) != self.tokenizer.eos_token_id
                )  # Utterance boundaries are not word boundaries
                word_starts[0] = 1  # First token is always a word start
                word_starts_list.append(word_starts)

        batch = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

        if self.get_word_boundaries:
            batch["word_starts"] = word_starts_list

        return batch


def load_from_text(text_path: Path) -> datasets.Dataset:
    """Create a dataset from the lines."""
    with text_path.open() as f:
        lines = [line.strip() for line in f if line.strip()]
    return datasets.Dataset.from_dict({"text": lines})


def load_dataset(train_path: Path, dev_path: Path) -> datasets.DatasetDict:
    """Loads dataset from text path."""
    train_dataset = load_from_text(train_path)
    val_dataset = load_from_text(dev_path)
    # Combine into a DatasetDict
    return datasets.DatasetDict(
        {
            "train": train_dataset["train"] if "train" in train_dataset else train_dataset,  # noqa: SIM401
            "valid": val_dataset["valid"] if "valid" in val_dataset else val_dataset,  # noqa: SIM401
        }
    )


def load_char_tokenizer() -> AutoTokenizer:
    """Load autokenizer from pre-trained."""
    return AutoTokenizer.from_pretrained("phonemetransformers/GPT2-85M-CHAR-TXT")
