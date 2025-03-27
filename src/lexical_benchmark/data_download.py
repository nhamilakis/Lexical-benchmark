from pathlib import Path

from lexical_benchmark import datasets, utils

_OBERON2_DATASETS_DIR = Path("/scratch1/projects/lexical-benchmark/v2/datasets")


def download_stela() -> None:
    """Download STELA Trascription dataset from coml/oberon2."""
    cfg: datasets.STELADatasetConfig = datasets.get_config("stela")

    cmd = utils.Rsync(
        remote_source="oberon2",
        source_dir=_OBERON2_DATASETS_DIR / cfg.root_dir.name,
        target_dir=cfg.root_dir,
        file_list=cfg.transfer_pathlist(),
        delete=True,
        copy_symlinks=True,
    )
    cmd(output_handling="log_debug")
