import dataclasses
import itertools
import typing as t
from pathlib import Path

from lexical_benchmark import datasets, exc, lb_types, settings
from lexical_benchmark.train.checkpoint_utils import GenerationCheckpoint

from .definitions import DatasetItemsLoader


@t.runtime_checkable
class _DatasetWithGenerations(t.Protocol):
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


class _CheckPointIteratorKwargs(t.TypedDict):
    """Optional Arguments for checkpoint iteration."""

    datasets: tuple[datasets.DATASET_NAMES, ...]
    model_types: tuple[lb_types.MODEL_TYPE, ...]
    temperatures: tuple[float, ...]
    hour_per_years: tuple[str, ...]
    langs: tuple[str, ...]
    splits: tuple[str, ...]
    chunks: tuple[str, ...]


@dataclasses.dataclass
class GenerationCheckpointLoader(DatasetItemsLoader):
    """Dataset loader for the checkpoint set.

    Example File Path:
    /<root-dir>/generation / checkpoints / stela3 / EN / 01 / 00 / lstm / checkpoint_0.6.obj
    """

    lang: str
    split: str
    chunk: str
    temperature: float
    model_type: lb_types.MODEL_TYPE
    dt_cfg: _DatasetWithGenerations

    def to_args_dict(self) -> dict[str, t.Any]:
        """Convert to dictionairy for args-index."""
        return {
            "dataset_name": self.dt_cfg.dataset_name,
            "model_type": self.model_type,
            "lang": self.lang,
            "split": self.split,
            "chunk": self.chunk,
            "resume": True,
            "override": False,
            "temperature": self.temperature,
            "completed": self.is_finished(),
        }

    @property
    def root_dir(self) -> Path:
        """Location to write to."""
        return (
            self.dt_cfg.generation_checkpoint_root
            / self.lang
            / f"{self.split:02}"
            / f"{self.chunk:02}"
            / self.model_type
        )

    @property
    def intermediate_checkpoint_file(self) -> Path:
        """Path to intermediate checkpoint file."""
        return self.root_dir / f"generation_{self.temperature}.intermediate.obj"

    @property
    def final_checkpoint_file(self) -> Path:
        """Path to final checkpoint."""
        return self.root_dir / f"generation_{self.temperature}.obj"

    @property
    def log_file(self) -> Path:
        """Path to the logfile."""
        return self.root_dir / f"generation_{self.temperature}.log"

    def load_intermediate(self) -> "GenerationCheckpoint | None":
        """Load intermediate checkpoint."""
        return GenerationCheckpoint.load_intermediate(
            location=self.root_dir,
            temperature=self.temperature,
        )

    def load_final(self) -> "GenerationCheckpoint | None":
        """Load final checkpoint."""
        return GenerationCheckpoint.load_final(
            location=self.root_dir,
            temperature=self.temperature,
        )

    def exists(self) -> bool:
        """Return if ressources exist."""
        return self.root_dir.exists()

    def has_intermediate(self) -> bool:
        """Return if generation has started."""
        return self.intermediate_checkpoint_file.is_file()

    def is_finished(self) -> bool:
        """Return if generation is completed."""
        if self.final_checkpoint_file.is_file():
            return True

        if self.has_intermediate():
            chk = self.load_intermediate()
            return chk.remaining_count() == 0
        return False

    def __post_init__(self) -> None:
        """post-creation checks."""
        # Fix padding for split & chunk
        self.split = f"{self.split:02}"
        self.chunk = f"{self.chunk:02}"

        # Check if correct dataset is provided.
        if not isinstance(self.dt_cfg, _DatasetWithGenerations):
            raise exc.DatasetTypeError(dataset=type(self.dt_cfg), protocol=_DatasetWithGenerations)

    @classmethod
    def load(
        cls,
        dataset_name: datasets.DATASET_NAMES,
        lang: str,
        split: str,
        chunk: str,
        temperature: float,
        model_type: lb_types.MODEL_TYPE,
    ) -> "GenerationCheckpointLoader":
        """Load item."""
        split = f"{split:02}"  # Make sure padding is properly applied
        chunk = f"{chunk:02}"  # Make sure padding is properly applied
        return cls(
            lang=lang,
            split=split,
            chunk=chunk,
            model_type=model_type,
            temperature=temperature,
            dt_cfg=datasets.get_config(dataset_name),
        )

    @classmethod
    def iter_items(cls, **kwargs: t.Unpack[_CheckPointIteratorKwargs]) -> t.Iterator["GenerationCheckpointLoader"]:
        """Iterate over a set of items."""
        dataset_list = kwargs.get("datasets", ("stela", "childes"))
        temperatures = kwargs.get("temperatures", settings.GENERATION_TEMPERATURES)
        model_type_list = kwargs.get("model_types", settings.MODEL_TYPES)

        for dt_name in dataset_list:
            dt_cfg: _DatasetWithGenerations = datasets.get_config(dt_name)
            langs_list = kwargs.get("langs", dt_cfg.langs)
            # Fix potential padding in chunk definitions
            split_list = tuple([f"{sp:02}" for sp in kwargs.get("splits", dt_cfg.size_splits)])
            chunk_list = tuple([f"{ck:02}" for ck in kwargs.get("chunks", settings.TRAIN_CHUNKS)])

            for _lang in langs_list:
                # Skip non-valid languages
                if _lang not in dt_cfg.langs:
                    continue

                for _split in split_list:
                    # Skip non-existing hours
                    if _split not in dt_cfg.size_splits:
                        continue

                    for _chunk in dt_cfg.chunks_by_size(_lang, _split):
                        # If a filter list is set keep only given chunks
                        if chunk_list and _chunk not in chunk_list:
                            continue

                        params_items = itertools.product(model_type_list, temperatures)
                        for _model, _temp in params_items:
                            yield cls.load(
                                dataset_name=dt_cfg.dataset_name,
                                lang=_lang,
                                split=_split,
                                chunk=_chunk,
                                model_type=_model,
                                temperature=_temp,
                            )


class _GenTextIteratorKwargs(t.TypedDict):
    """Optional Arguments for checkpoint iteration."""

    datasets: tuple[datasets.DATASET_NAMES, ...]
    model_types: tuple[lb_types.MODEL_TYPE, ...]
    langs: tuple[str, ...]
    temperatures: tuple[float, ...]
    estimation_types: tuple[lb_types.ESTIMATION_TYPE, ...]
    month_include: tuple[int, ...]
    temperatures: tuple[float, ...]


@dataclasses.dataclass
class GenerationItemsLoader(DatasetItemsLoader):
    """Dataset loader for the generated text architecture.

    Example File Path:
    /<root-dir> / generation / text / stela3 / EN /  07 / lstm / 100hpy / 01_00_0.6.txt
    """

    estimation_type: lb_types.ESTIMATION_TYPE
    model_type: lb_types.MODEL_TYPE
    lang: str
    month: int
    model_chunk: str
    temperature: float
    dt_cfg: _DatasetWithGenerations

    @property
    def root_dir(self) -> Path:
        """Path to the root directory."""
        return (
            self.dt_cfg.generation_text_root / self.lang / f"{self.month:02}" / self.model_type / self.estimation_type
        )

    @property
    def text_file(self) -> Path:
        """Path to the text file."""
        return self.root_dir / f"{self.model_chunk}_{self.temperature}.txt"

    @classmethod
    def load(
        cls,
        dataset_name: datasets.DATASET_NAMES,
        model_type: lb_types.MODEL_TYPE,
        estimation_type: lb_types.ESTIMATION_TYPE,
        lang: str,
        model_chunk: str,
        month: int,
        temperature: float,
    ) -> "GenerationItemsLoader":
        """Load specific item."""
        return cls(
            dt_cfg=datasets.get_config(dataset_name),
            estimation_type=estimation_type,
            model_type=model_type,
            lang=lang,
            model_chunk=model_chunk,
            month=month,
            temperature=temperature,
        )

    @classmethod
    def iter_items(cls, **kwargs: t.Unpack[_GenTextIteratorKwargs]) -> t.Iterable["GenerationItemsLoader"]:
        """Iterate over checkpoint items."""
        dataset_list = kwargs.get("datasets", ("stela", "child_realistic"))
        temperatures = kwargs.get("temperatures", settings.GENERATION_TEMPERATURES)
        model_type_list = kwargs.get("model_types", settings.MODEL_TYPES)
        estimation_list = kwargs.get("estimation_types", settings.MONTH_ESTIMATES)
        default_months = tuple(range(settings.MONTH_RANGE[0], settings.MONTH_RANGE[1] + 1))
        month_list = kwargs.get("month_include", default_months)
        chunk_list = kwargs.get("chunks", settings.TRAIN_CHUNKS)

        params_items = itertools.product(
            temperatures,  # 0
            model_type_list,  # 1
            estimation_list,  # 2
            month_list,  # 3
        )

        for dt_name in dataset_list:
            dt_cfg: _DatasetWithGenerations = datasets.get_config(dt_name)
            langs_list = kwargs.get("langs", dt_cfg.langs)
            split_list = kwargs.get("splits", dt_cfg.size_splits)
            chunk_list = kwargs.get("chunks", settings.TRAIN_CHUNKS)

            for _lang in langs_list:
                # Skip non-valid languages
                if _lang not in dt_cfg.langs:
                    continue
                for _split in split_list:
                    # Skip non-existing hours
                    if _split not in dt_cfg.size_splits:
                        continue

                    for _chunk in dt_cfg.chunks_by_size(_lang, _split):
                        # If a filter list is set keep only given chunks
                        if len(chunk_list) != 0 and _chunk not in chunk_list:
                            continue

                        for params in params_items:
                            yield cls.load(
                                dataset_name=dt_name,
                                temperature=params[0],
                                model_type=params[1],
                                estimation_type=params[2],
                                month=params[3],
                                lang=_lang,
                                model_chunk=_chunk,
                            )
