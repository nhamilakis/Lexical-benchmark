from pathlib import Path

from lexical_benchmark import datasets, utils


def upload_stela(host: str, target_dir: Path) -> None:
    """Upload the stela dataset to a target directory on a remote host."""
    cfg: datasets.STELADatasetConfig = datasets.get_config("stela")
    cmd = utils.Rsync(
        source_dir=cfg.root_dir,
        target_dir=target_dir / cfg.root_dir.name,
        remote_dest=host,
        file_list=cfg.transfer_pathlist(),
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
        file_list=cfg.transfer_pathlist(),
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
