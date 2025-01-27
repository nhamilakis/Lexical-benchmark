#!/usr/bin/env python
import argparse
import string
from pathlib import Path

from lexical_benchmark import settings
from lexical_benchmark.utils.hf_util import HFCharacterTokenizer


def parseargs() -> argparse.Namespace:
    """Argument parser from CMD."""
    parser = argparse.ArgumentParser(description="save the tokenizer to the relative path")
    parser.add_argument(
        "--tokenizer_path", 
        default="models/tokenizer", 
        help="Path to the validation file")
    parser.add_argument(
        "--added_tokens", 
        default=["'", "|"], 
        help="A list of added special tokens"
        )
    return parser.parse_args()




def main():
    """Main training function."""
    args = parseargs()
    tokenizer_dir:Path = settings.PATH.DATA_DIR / args.tokenizer_path

    # Initialize tokenizer
    tokenizer = HFCharacterTokenizer(chars=string.ascii_letters, model_max_length=1024)
    # Add any special tokens if needed
    tokenizer.add_tokens(args.added_tokens)
    # Save the tokenizer
    tokenizer.save_pretrained(str(tokenizer_dir))
    print(f"Saving the tokenizer to {tokenizer_dir}")



if __name__ == "__main__":
    main()
