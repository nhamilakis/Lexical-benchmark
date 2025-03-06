from pathlib import Path

from lexical_benchmark import datasets, utils


def upload_stela(host: str, target_dir: Path) -> None:
    """Upload the stela dataset to a target directory on a remote host."""
    cfg: datasets.STELADatasetConfig = datasets.get_config("stela")
    cmd = utils.Rsync(
        source_dir=cfg.root_dir,
        target_dir=target_dir / cfg.root_dir.name,
        remote_dest=host,
        file_list=[
            cfg.preprocessed_root.relative_to(cfg.root_dir),  # src/preprocess
            cfg.by_hour_dir.relative_to(cfg.root_dir),  # by_hour/
            cfg.meta_dir.relative_to(cfg.root_dir),  # metadata/
            cfg.by_size_dir.relative_to(cfg.root_dir),  # by_size/
            cfg.by_genre_dir.relative_to(cfg.root_dir),  # by_genre/
        ],
        delete=True,
        copy_symlinks=True,
    )
    cmd(output_handling="log_info")


def upload_childes(host: str, target_dir: Path) -> None:
    """Upload the stela dataset to a target directory on a remote host."""
    cfg: datasets.CHILDESDatasetConfig = datasets.get_config("childes")
    cmd = utils.Rsync(
        source_dir=cfg.root_dir,
        target_dir=target_dir / cfg.root_dir.name,
        remote_dest=host,
        file_list=[
            cfg.preprocessed_root.relative_to(cfg.root_dir),  # src/preprocess
            cfg.meta_dir.relative_to(cfg.root_dir),  # metadata/
            # TODO: add rest of CHILDES
        ],
        delete=True,
        copy_symlinks=True,
    )
    cmd(output_handling="log_info")


def upload_childrealistic(host: str, target_dir: Path) -> None:
    """Upload the stela dataset to a target directory on a remote host."""
    cfg: datasets.ChildRealisticDatasetConfig = datasets.get_config("child_realistic")
    cmd = utils.Rsync(
        source_dir=cfg.root_dir,
        target_dir=target_dir / cfg.root_dir.name,
        remote_dest=host,
        delete=True,
        copy_symlinks=True,
    )
    cmd(output_handling="log_info")
