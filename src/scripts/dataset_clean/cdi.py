#!/usr/bin/env python
"""Building Word-CDI Dataset."""

from pathlib import Path

from lexical_benchmark import settings
from lexical_benchmark.datasets import wordbank_cdi
from lexical_benchmark.utils import slurm_utils

slurm_utils.info_header()

source = Path.cwd() / "data-v2/datasets/wordbank-cdi/src/original/ENG-NA/WG/cdi-produce.csv"
target = Path.cwd() / "data-v2/datasets/wordbank-cdi/en-na/wg_cdi_produce.csv"
try:
    age_min, age_max = settings.WORDBANK_CDI.age_range(lang="ENG-NA", form="WG")  # type: ignore
except TypeError as e:
    raise ValueError("Age range not valid !") from e

print("Loading CDI, and performing cleanup tasks !!", flush=True)
cdi_prep = wordbank_cdi.CDIPreparation(age_min=age_min, age_max=age_max, raw_csv=source)
cdi_prep.load_dataset()
df = cdi_prep.build_gold(
    do_type_filtering=False,
    filter_item_definitions=False,
    filter_pos_categories=False,
)
target.parent.mkdir(exist_ok=True, parents=True)
print(f"Saving result to {target} !!", flush=True)
df.to_csv(target, index=False)

# Job Done
slurm_utils.info_footer()
