#!/usr/bin/env python
from pathlib import Path

import polars as pl

from lexical_benchmark.datasets import utils as dataset_utils

file = Path("data/HSLLD_HV5_LW_jerlw5.processed")

print("Loading model !")
pos_model = dataset_utils.various.spacy_model("en_core_web_trf", require_gpu=True)
lines = file.read_tokenized()
print("Runnig NLP analysis...")
results = list(pos_model.pipe(lines, batch_size=1024))
print("Extracting results...")

items = []
for line in results:
    items.extend([(token.text, token.lemma_, token.pos_, token.tag_) for token in line])


df = pl.DataFrame(items, schema=["text", "lemma", "pos", "tag"])
df.write_csv(Path("data/post-test-by_word.csv"))

print("Done !")
