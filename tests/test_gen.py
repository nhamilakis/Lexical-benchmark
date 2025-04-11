import typing as t
from pathlib import Path

import torch
from transformers import AutoTokenizer

from lexical_benchmark.train.generators.batch_generator import BatchGenerator

Model = t.Any

model_root = Path("/scratch1/projects/lexical-benchmark/v2/models/stela3/EN/01/00")
model_type = "gpt2"
model_path = model_root / model_type

tokenizer_name = "phonemetransformers/GPT2-85M-CHAR-TXT"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
use_vllm = True
temp = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"
nb_tokens = 10

batch_generator = BatchGenerator(
    model_path=model_path,
    tokenizer=tokenizer,
    device=device,
    use_vllm=use_vllm,
    model_type=model_type,
    temp=temp,
    nb_tokens=nb_tokens,
)

generated_text = batch_generator.generate_text()
print(generated_text)
