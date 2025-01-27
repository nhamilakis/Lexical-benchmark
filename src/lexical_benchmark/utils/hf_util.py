import json
import os
import string
from typing import Dict, List, Tuple, Union

from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING


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


def load_char_tokenizer(model_max_length: int = 2048, special_token_lst: list[str] = ["'", "|"]):
    # config and load the tokenizer
    chars = string.ascii_letters
    tokenizer = CharacterTokenizer(chars, model_max_length)
    # add the new token for abbreviation
    for special_token in special_token_lst:
        tokenizer.add_tokens(special_token)
    return tokenizer


class HFCharacterTokenizer(PreTrainedTokenizer):
    """Character-level tokenizer compatible with Hugging Face."""

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, chars: str = None, model_max_length: int = 1024, **kwargs) -> None:
        """Initialize tokenizer."""
        # Initialize vocabulary before parent class
        # Initialize with special tokens
        special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

        # If loading from saved, chars might be None
        self.chars = list(chars) + special_tokens if chars else special_tokens

        # Create vocabulary
        self.vocab = {char: idx for idx, char in enumerate(self.chars)}
        self.decoder = {idx: char for char, idx in self.vocab.items()}

        # Set special tokens
        self._unk_token = "[UNK]"
        self._eos_token = "[EOS]"
        self._pad_token = "[PAD]"
        self._bos_token = "[BOS]"

        # Initialize parent class after vocabulary is set up
        super().__init__(model_max_length=model_max_length, **kwargs)

    @property
    def vocab_size(self) -> int:
        """Return size of vocabulary."""
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Return vocabulary dictionary."""
        return self.vocab.copy()

    def _tokenize(self, text: str) -> List[str]:
        """Convert text to list of characters."""
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to vocabulary id."""
        return self.vocab.get(token, self.vocab[self._unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        """Convert vocabulary id to token."""
        return self.decoder.get(index, self._unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens back to string."""
        return "".join(tokens)

    def save_pretrained(self, save_directory: str, **kwargs) -> Tuple[str, ...]:
        """Save the tokenizer and its configuration to a directory."""
        os.makedirs(save_directory, exist_ok=True)

        # Save special tokens config
        special_tokens_map = {
            "unk_token": self._unk_token,
            "eos_token": self._eos_token,
            "pad_token": self._pad_token,
            "bos_token": self._bos_token,
        }
        special_tokens_map_file = os.path.join(save_directory, "special_tokens_map.json")
        with open(special_tokens_map_file, "w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f, ensure_ascii=False)

        # Save tokenizer config
        tokenizer_config = {
            "model_max_length": self.model_max_length,
            "chars": self.chars,
            "tokenizer_class": "CharacterTokenizer",
        }
        tokenizer_config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(tokenizer_config_file, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, ensure_ascii=False)

        # Save vocabulary
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

        return (special_tokens_map_file, tokenizer_config_file, vocab_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
        """Load tokenizer from saved files."""
        # Load tokenizer config
        config_file = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Load vocabulary
        vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        # Initialize tokenizer
        chars = [char for char in config["chars"] if char not in ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]]
        instance = cls(chars="".join(chars), model_max_length=config["model_max_length"], **kwargs)

        # Restore vocab if it differs from initialization
        instance.vocab = {k: int(v) for k, v in vocab.items()}
        instance.decoder = {int(v): k for k, v in vocab.items()}
        instance.chars = config["chars"]

        return instance


# Register the tokenizer
AutoTokenizer.register(HFCharacterTokenizer, "HFCharacterTokenizer")

# Add to TOKENIZER_MAPPING if needed
TOKENIZER_MAPPING.register(AutoConfig, (HFCharacterTokenizer, None))
