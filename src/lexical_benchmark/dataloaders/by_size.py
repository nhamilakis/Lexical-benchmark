import dataclasses
import typing as t
from pathlib import Path

from lexical_benchmark import datasets, exc, lb_types, settings
from lexical_benchmark.text_lib import tokenization

from .definitions import DatasetItemsLoader


@t.runtime_checkable
class DatasetWithBySize(t.Protocol):
    """Dataset config with support for by_chunk split."""

    @property
    def dataset_name(self) -> str:
        """Name of the dataset."""
        ...

    @property
    def generation_checkpoint_root(self) -> Path:
        """Root location for checkpoint of generations."""
        ...

    @property
    def generation_text_root(self) -> Path:
        """Root location for generated text."""
        ...

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


class BySizeTrainStruct(t.TypedDict):
    """DataStructure to export TrainArgs."""

    model_type: lb_types.MODEL_TYPE
    dataset_name: str
    lang: str
    split: str
    chunk: str
    completed_training: bool
    last_checkpoint: Path | None
    resume: bool = True
    override: bool = False
    resume_id: str | None = None


@dataclasses.dataclass
class BySizeItemsLoader(DatasetItemsLoader):
    """Dataset loader for by_size architecture."""

    lang: str
    split: str
    chunk: str
    dt_cfg: DatasetWithBySize

    def __post_init__(self) -> None:
        # Check if correct dataset is provided.
        if not isinstance(self.dt_cfg, DatasetWithBySize):
            raise exc.DatasetTypeError(dataset=type(self.dt_cfg), protocol=DatasetWithBySize)

    @classmethod
    def load(cls, dataset_name: datasets.DATASET_NAMES, lang: str, split: str, chunk: str) -> "DatasetItemsLoader":
        """Load item directly."""
        split = f"{split:02}"  # Make sure padding is properly applied
        chunk = f"{chunk:02}"  # Make sure padding is properly applied

        return cls(lang=lang, split=split, chunk=chunk, dt_cfg=datasets.get_config(dataset_name))

    @property
    def chunk_id(self) -> str:
        """Build the id of the current chunk."""
        return f"{self.lang}_{self.split}_{self.chunk}"

    @property
    def data_dir(self) -> str:
        """Current chunk path."""
        return self.dt_cfg.by_size_dir / self.lang / self.split / self.chunk

    @property
    def model_root(self) -> Path:
        """Root directory to the corresponding model folder."""
        return settings.PATH.model_root / self.dt_cfg.dataset_name / self.lang / self.split / self.chunk

    @property
    def geneneration_checkpoint_root(self) -> Path:
        """Root directory to the generated data is raw (checkpoint form)."""
        return self.dt_cfg.generation_checkpoint_root / self.lang / self.split / self.chunk

    @property
    def train_file(self) -> Path:
        """Transcription file."""
        return self.data_dir / "train.txt"

    @property
    def dev_file(self) -> Path:
        """Path to dev set."""
        return self.dt_cfg.by_size_dir / self.lang / "dev" / "dev.txt"

    def tokenized_train(self, *, as_path: bool = False) -> list[str] | Path:
        """Load train trainscription in tokenized form."""
        file = self.train_file.parent / f"{self.train_file.stem}.tokenized"

        if file.is_file() and not as_path:
            return file.safe_readlines()

        if file.is_file() and as_path:
            return file

        untokenized_txt = self.train_file.safe_readlines()
        tokenized_text = [tokenization.hf_line_format(line) for line in untokenized_txt]

        # Save to file
        file.write_text("\n".join(tokenized_text))

        if as_path:
            return file
        return tokenized_text

    def tokenized_dev(self, *, as_path: bool = False) -> list[str] | Path:
        """Load dev transcription in tokenized form."""
        file = self.dev_file.parent / f"{self.dev_file.stem}.tokenized"

        if file.is_file() and not as_path:
            return file.safe_readlines()
        if file.is_file() and as_path:
            return file

        untokenized_txt = self.dev_file.safe_readlines()
        tokenized_text = [tokenization.hf_line_format(line) for line in untokenized_txt]

        # Save to file
        file.write_text("\n".join(tokenized_text))

        if as_path:
            return file
        return tokenized_text

    def train_args(
        self,
        *,
        model_type: lb_types.MODEL_TYPE,
        resume: bool = True,
        override: bool = False,
        resume_id: str | None = None,
    ) -> "BySizeTrainItem":
        """Load train arguments."""
        return BySizeTrainItem(
            model_type=model_type,
            data_item=self,
            resume=resume,
            override=override,
            resume_id=resume_id,
        )

    @classmethod
    def iter_items(cls, dataset_name: datasets.DATASET_NAMES, **kwargs) -> t.Iterable["BySizeItemsLoader"]:
        """Iterate over by_size items.

        Raises:
            exc.UnknownDatasetNameError: if dataset config does not exist.

        """
        cfg: DatasetWithBySize = datasets.get_config(dataset_name)
        langs_list = kwargs.get("langs", cfg.langs)
        split_list = kwargs.get("splits", cfg.size_splits)
        # Fix formatting issues
        split_list = [f"{sp:02}" for sp in split_list]
        chunk_list = kwargs.get("chunks", ())
        chunk_list = [f"{ck:02}" for ck in chunk_list]

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
                    yield cls.load(dataset_name=dataset_name, lang=_lang, split=_split, chunk=_chunk)


@dataclasses.dataclass
class BySizeTrainItem:
    """Structure allowing inferring training arguments."""

    model_type: lb_types.MODEL_TYPE
    data_item: BySizeItemsLoader
    resume: bool = True
    override: bool = False
    resume_id: str | None = None

    @property
    def item_id(self) -> str:
        """Unique identifier for item."""
        return f"{self.data_item.lang}_{self.data_item.split}_{self.data_item.chunk}"

    @property
    def job_name(self) -> str:
        """Unique train item identifier."""
        return f"{self.dataset_name}_{self.model_type}_{self.item_id}"

    @property
    def dataset_name(self) -> str:
        """Name of the dataset."""
        return self.data_item.dt_cfg.dataset_name

    @property
    def model_root_dir(self) -> Path:
        """Root directory."""
        return self.data_item.model_root / self.model_type

    @property
    def train_logs_file(self) -> Path:
        """Path to file containing training logs."""
        return self.model_root_dir / "training.logs"

    @property
    def generation_root_dir(self) -> Path:
        """Root directory."""
        return self.data_item.geneneration_checkpoint_root / self.model_type

    @property
    def completed_training(self) -> bool:
        """Check if completed."""
        return (self.model_root_dir / "training_args.bin").is_file()

    def get_resume_train(self) -> Path | None:
        """Conditional function that checks if it is required to resume training."""
        match (self.override, self.resume, self.resume_id):
            case (True, _, _):  # When overriding always restart
                return None
            case (False, True, None):  # Resume from last
                return self.last_checkpoint()
            case (False, True, _):  # Resume from specific
                return self.get_checkpoint(self.resume_id)
            case _:  # Start from scratch
                if len(self.checkpoint_list()) > 0:
                    raise ValueError("Trying to override model-root when 'override=False' !!")
                return None

    def checkpoint_list(self, *, sort: bool = True) -> list[Path]:
        """List of checkpoints."""
        if not self.model_root_dir.is_dir():
            return []
        # checkpoint folders are named 'checkpoint-XXXX' where XXXX is a number
        list_of_checkpoint_files = [
            d for d in self.model_root_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        if sort:
            # reverse=true grabs the largest number (assumption: biggest=latest)
            return sorted(
                list_of_checkpoint_files, key=lambda path: int(path.name.replace("checkpoint-", "")), reverse=True
            )
        return list_of_checkpoint_files

    def get_checkpoint(self, check_id: str) -> Path | None:
        """Get a specific checkpoint if it exists.."""
        for chk in self.checkpoint_list():
            if chk.name == f"checkpoint-{check_id}":
                return chk
        return None

    def last_checkpoint(self) -> Path | None:
        """Get location of latest checkpoint."""
        return next(
            iter(self.checkpoint_list(sort=True)),
            None,
        )

    def train_txt(self) -> Path:
        """Load train data."""
        return self.data_item.tokenized_train(as_path=True)

    def dev_txt(self) -> Path:
        """Load dev data."""
        return self.data_item.tokenized_dev(as_path=True)

    def to_dict(self) -> BySizeTrainStruct:
        """Convert item into a dictionairy."""
        last_checkpoint = self.last_checkpoint()
        return {
            "model_type": self.model_type,
            "lang": self.data_item.lang,
            "split": self.data_item.split,
            "chunk": self.data_item.chunk,
            "dataset_name": self.dataset_name,
            "completed_training": self.completed_training,
            "generation_root": str(self.generation_root_dir) if self.generation_root_dir else self.generation_root_dir,
            "model_root": str(self.model_root_dir) if self.model_root_dir else self.model_root_dir,
            "last_checkpoint": str(last_checkpoint) if last_checkpoint else last_checkpoint,
            "resume": self.resume,
            "override": self.override,
            "resume_id": self.resume_id,
        }

    @classmethod
    def from_dict(cls, cfg_args: BySizeTrainStruct) -> "BySizeTrainItem":
        """Load train args from a dictionairy."""
        return cls(
            model_type=cfg_args["model_type"],
            data_item=BySizeItemsLoader(
                lang=cfg_args["lang"],
                split=cfg_args["split"],
                chunk=cfg_args["split"],
            ),
        )
