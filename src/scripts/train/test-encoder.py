#!/usr/bin/python

from pathlib import Path

import IPython
from rich.console import Console
from transformers import (
    AutoTokenizer,
)

from lexical_benchmark.text_lib.tokenization import hf_file_format
from lexical_benchmark.train import tokenizers

console = Console()
tokenizer = AutoTokenizer.from_pretrained("phonemetransformers/GPT2-85M-CHAR-TXT")
file = Path("~/workspace/data/test.txt").expanduser()
file_stela = Path("/scratch1/projects/lexical-benchmark/v2/datasets/stela3/by_size/EN/01/00/train.tokenized")


hf_file_format(file)
file_w_boundaries = file.parent / f"{file.stem}.tokenized.hf"

tokenized_text = tokenizers.load_joined_text(file_w_boundaries, tokenizer=tokenizer, max_length=30)
print(f"Training dataset size: {len(tokenized_text)}")
console.print(tokenized_text)
console.print(tokenized_text.features)
console.print(tokenized_text.features["input_ids"].feature)
IPython.embed()
