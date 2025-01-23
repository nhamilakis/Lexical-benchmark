from pathlib import Path

from lexical_benchmark import settings
from lexical_benchmark.datasets.gen_child.preparation import CHILDESMonth

root_dir: Path = settings.PATH.dataset_root / "CHILDES"
out_dir: Path = settings.PATH.DATA_DIR  /"gen"/ "test"


processor = CHILDESMonth(root_dir=root_dir,
                             out_dir=settings.PATH.DATA_DIR / out_dir)
processor.process()
