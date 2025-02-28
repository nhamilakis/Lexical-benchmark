from pathlib import Path

import pandas as pd

from lexical_benchmark.utils import gen_util

MODEL_PATH = Path("/lustre/fswork/projects/rech/hhb/ucx81cx/data/models") / "ChildRealistic/by_month/EN/30/00/trans"


# Create generator
generator = gen_util.TextGenerator(model_path=str(MODEL_PATH), model_type="transformer", use_vllm=True)

processor = gen_util.BatchProcessor(
    generator=generator,
    save_path=Path.cwd(),
    chunk_size=2,
    hour_per_year=1000,
    debug=True,
)


sub_batch = pd.DataFrame([[1], [2], [3], [4], [5]], columns=["sent_len"])
res = processor._process_subbatch(
    sub_batch=sub_batch,
    temp_lst=[0.3, 0.6, 1.0, 1.5],
    temp_columns=["0.3", "0.6", "1.0", "1.5"],
)
