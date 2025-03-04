import dataclasses
import typing as t
from pathlib import Path

from lexical_benchmark import datasets
from lexical_benchmark.text_lib import tokenization

from .definitions import DatasetItemsLoader


class DatasetWithBySize(t.Protocol):
    """Dataset config with support for by_chunk split."""

    @property
    def by_size_dir(self) -> Path:
        """Path to by_size section of the dataset."""
        ...

    @property
    def langs(self) -> tuple[str, ...]:
        """Available languages."""
        ...

    @property
    def size_splits(self) -> tuple[str, ...]:
        """Available by_size splits."""
        ...

    def chunks_by_size(self, lang: str, split: str) -> tuple[str, ...]:
        """Available chunks in a split in the by_size architecture."""
        ...


@dataclasses.dataclass
class BySizeItemsLoader(DatasetItemsLoader):
    """Dataset loader for by_size architecture."""

    lang: str
    split: str
    chunk: str
    dt_cfg: DatasetWithBySize

    @property
    def chunk_id(self) -> str:
        """Build the id of the current chunk."""
        return f"{self.split}_{self.chunk}"

    @property
    def root_dir(self) -> str:
        """Current chunk path."""
        return self.dt_cfg.by_size_dir / self.lang / self.split / self.chunk

    @property
    def train_file(self) -> Path:
        """Transcription file."""
        return self.root_dir / "transcription.txt"

    @property
    def dev_file(self) -> Path:
        """Path to dev set."""
        # TODO: figure out if the file local or global
        # dt_cfg.by_size_dir / self.lang / dev / transcription.txt
        return self.root_dir / "dev.txt"

    def tokenized_train(self) -> list[str]:
        """Load train trainscription in tokenized form."""
        file = self.train_file.parent / f"{self.train_file.stem}.tokenized"

        if file.is_file():
            return file.safe_readlines()

        untokenized_txt = self.train_file.safe_readlines()
        tokenized_text = [tokenization.hf_line_format(line) for line in untokenized_txt]

        # Save to file
        file.write_text("\n".join(tokenized_text))
        return tokenized_text

    def tokenized_dev(self) -> list[str]:
        """Load dev transcription in tokenized form."""
        file = self.dev_file.parent / f"{self.dev_file.stem}.tokenized"

        if file.is_file():
            return file.safe_readlines()

        untokenized_txt = self.dev_file.safe_readlines()
        tokenized_text = [tokenization.hf_line_format(line) for line in untokenized_txt]

        # Save to file
        file.write_text("\n".join(tokenized_text))
        return tokenized_text

    @classmethod
    def iter_items(cls, dataset_name: datasets.DATASET_NAMES, **kwargs) -> t.Iterable["BySizeItemsLoader"]:
        """Iterate over by_size items.

        Raises:
            exc.UnknownDatasetNameError: if dataset config does not exist.

        """
        cfg: DatasetWithBySize = datasets.get_config(dataset_name)
        langs_list = kwargs.get("langs", cfg.langs)
        split_list = kwargs.get("splits", cfg.size_splits)
        chunk_list = kwargs.get("chunks", ())

        for _lang in langs_list:
            # Skip non-valid languages
            if _lang not in cfg.langs:
                continue

            for _split in split_list:
                # Skip non-existing hours
                if _split not in cfg.size_splits:
                    continue

                for _chunk in cfg.chunks_by_size(_lang, _split):
                    # If a filter list is set keep only given chunks
                    if len(chunk_list) != 0 and _chunk not in chunk_list:
                        continue

                    # Build item
                    yield cls(lang=_lang, split=_split, chunk=_chunk, dt_cfg=cfg)
