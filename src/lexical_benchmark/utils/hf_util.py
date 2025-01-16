import string

from transformers import LineByLineTextDataset, PreTrainedTokenizer


class CharacterTokenizer(PreTrainedTokenizer):
    """Tokenization for characters."""

    def __init__(self, chars: str, model_max_length: int) -> None:
        self.model_max_length = model_max_length

        # Initialize with special tokens
        special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        self.chars = list(chars) + special_tokens

        # Create vocabulary
        self.vocab = {char: idx for idx, char in enumerate(self.chars)}
        self.decoder = {idx: char for char, idx in self.vocab.items()}

        # Set special tokens
        self.unk_token = "[UNK]"
        self.eos_token = "[EOS]"
        self.pad_token = "[PAD]"
        self.bos_token = "[BOS]"
        super().__init__()

    def add_tokens(self, new_tokens: str) -> int:
        """Add new tokens to the vocabulary."""
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]

        tokens_added = 0
        for token in new_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.decoder[len(self.decoder)] = token
                self.chars.append(token)
                tokens_added += 1
        return tokens_added

    def get_vocab(self) -> dict[str, int]:
        """Return vocabulary."""
        return self.vocab.copy()

    def _tokenize(self, text: str) -> list[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def decode(self, token_ids, skip_special_tokens=True, **kwargs) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        tokens = [self._convert_id_to_token(id_) for id_ in token_ids]

        if skip_special_tokens:
            tokens = [token for token in tokens if token not in [self.pad_token, self.eos_token, self.bos_token]]

        return self.convert_tokens_to_string(tokens)


def load_char_tokenizer(model_max_length: int=2048, special_token_lst: list[str]=["'", "|"]):
    # config and load the tokenizer
    chars = string.ascii_letters
    tokenizer = CharacterTokenizer(chars, model_max_length)
    # add the new token for abbreviation
    for special_token in special_token_lst:
        tokenizer.add_tokens(special_token)
    """
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'  # Set padding to the left side
        """
    return tokenizer


def tokenize_data(tokenizer, data_path, block_size: int):
    """Tokenize the dataset."""
    return LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=data_path,
        block_size=block_size,
    )
