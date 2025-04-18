from pathlib import Path

import torch

from lexical_benchmark.train.generation import BatchGenerator

model_root = Path("/scratch1/projects/lexical-benchmark/v2/models/stela3/EN/01/00/")
tokenizer_name = "phonemetransformers/GPT2-85M-CHAR-TXT"


# loop over different models
model_type = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = model_root / model_type
use_vllm = model_type == "gpt2"
nb_tokens = 100
batch_size = 1
temperature = 1.0


gen_attrs = {("100hpy", 6): 300, ("100hpy", 7): 250, ("100hpy", 8): 170, ("100hpy", 9): 120}


generator = BatchGenerator(
    model_path=model_path,
    tokenizer_name=tokenizer_name,
    device=device,
    use_vllm=use_vllm,
    model_type=model_type,
    batch_size=batch_size,  # New parameter to control batch size
)

text = generator.save_generation(temperature=temperature, gen_attrs=gen_attrs)
print(text)
