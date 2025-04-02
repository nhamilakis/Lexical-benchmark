import logging
import typing as t
from pathlib import Path

from clypi import Command, Positional

from lexical_benchmark import datasets, settings, utils

_REMOTE_DATASET_LOCATIONS = {
    "oberon2": Path("/scratch1/projects/lexical-benchmark/v2/datasets"),
    "jean-zay": Path("/lustre/fswork/projects/rech/hhb/commun/lexical-benchmark/datasets"),
}
_REMOTE_MODEL_LOCATIONS = {
    "oberon2": Path("/scratch1/projects/lexical-benchmark/v2/models"),
    "jean-zay": Path("/lustre/fsn1/projects/rech/hhb/commun/lexical-benchmark/models"),
}
_REMOTE_LIGHT_MODEL_LOCATIONS = {
    "oberon2": Path("/scratch1/projects/lexical-benchmark/v2/models-light"),
    "jean-zay": Path("/lustre/fsn1/projects/rech/hhb/commun/lexical-benchmark/models-light"),
}
_REMOTE_GENERATION_LOCATIONS = {
    "oberon2": Path("/scratch1/projects/lexical-benchmark/v2/generation"),
    "jean-zay": Path("/lustre/fsn1/projects/rech/hhb/commun/lexical-benchmark/generation"),
}


SourceTypes = t.Literal["oberon2", "jean-zay"]
L = logging.getLogger(__name__)


class Datasets(Command):
    """Dataset Downloader."""

    dataset: Positional[datasets.DATASET_NAMES]
    source: SourceTypes = "oberon2"
    dry_run: bool = False
    ignore_errors: bool = False
    output_handling: t.Literal["log_info", "log_debug", "print", "ignore"] = "log_info"
    logging: utils.generic.LOG_LEVELS = "INFO"

    def download_command(self) -> utils.Rsync:
        """Download one of the registered datasets from one of the servers."""
        cfg = datasets.get_config(self.dataset)
        remote_root = _REMOTE_DATASET_LOCATIONS.get(self.source)
        target_dir = settings.PATH.dataset_root / cfg.dataset_name

        return utils.Rsync(
            remote_source=self.source,
            source_dir=remote_root / cfg.root_dir.name,
            target_dir=target_dir,
            file_list=cfg.transfer_pathlist(),
            delete=True,
            copy_symlinks=True,
        )

    async def run(self) -> None:
        """Command runner."""
        utils.generic.setup_logging(self.logging)
        cmd = self.download_command()
        cmd(dry_run=self.dry_run, ignore_errors=self.ignore_errors, output_handling=self.output_handling)


class Models(Command):
    """Model Downloader."""

    dataset: datasets.DATASET_NAMES
    light_version: bool = True
    source: SourceTypes = "oberon2"
    dry_run: bool = False
    ignore_errors: bool = False
    output_handling: t.Literal["log_info", "log_debug", "print", "ignore"] = "log_info"
    logging: utils.generic.LOG_LEVELS = "INFO"

    def download_command(self) -> utils.Rsync:
        """Download models."""
        cfg = datasets.get_config(self.dataset)
        if self.light_version:
            remote_root = _REMOTE_LIGHT_MODEL_LOCATIONS.get(self.source)
            target_dir = settings.PATH.model_light_root
        else:
            remote_root = _REMOTE_MODEL_LOCATIONS.get(self.source)
            target_dir = settings.PATH.model_root

        return utils.Rsync(
            remote_source=self.source,
            source_dir=remote_root / cfg.dataset_name,
            target_dir=target_dir / cfg.dataset_name,
            delete=True,
            copy_symlinks=True,
        )

    async def run(self) -> None:
        """Command runner."""
        utils.generic.setup_logging(self.logging)
        cmd = self.download_command()
        cmd(dry_run=self.dry_run, ignore_errors=self.ignore_errors, output_handling=self.output_handling)


class Generation(Command):
    """Generation Folders Downloader."""

    dataset: datasets.DATASET_NAMES
    source: SourceTypes = "oberon2"
    dry_run: bool = False
    ignore_errors: bool = False
    output_handling: t.Literal["log_info", "log_debug", "print", "ignore"] = "log_info"
    logging: utils.generic.LOG_LEVELS = "INFO"

    def download_command(self) -> utils.Rsync:
        """Download one of the registered datasets from one of the servers."""
        cfg = datasets.get_config(self.dataset)
        remote_root = _REMOTE_GENERATION_LOCATIONS.get(self.source)
        target_dir = settings.PATH.generate_root

        return utils.Rsync(
            remote_source=self.source,
            source_dir=remote_root / cfg.dataset_name,
            target_dir=target_dir / cfg.dataset_name,
            delete=True,
            copy_symlinks=True,
        )

    async def run(self) -> None:
        """Command runner."""
        utils.generic.setup_logging(self.logging)
        cmd = self.download_command()
        cmd(dry_run=self.dry_run, ignore_errors=self.ignore_errors, output_handling=self.output_handling)


class Downloader(Command):
    """Command line data downloader."""

    subcommand: Datasets | Models | Generation

    async def run(self) -> None:
        """Command runner."""


def entrypoint() -> None:
    """Entrypoint."""
    cmd = Downloader.parse()
    cmd.start()


if __name__ == "__main__":
    entrypoint()
