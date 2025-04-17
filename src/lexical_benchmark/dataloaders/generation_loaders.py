import dataclasses
import typing as t
from pathlib import Path

from lexical_benchmark import datasets, exc, lb_types, settings
from lexical_benchmark.train.checkpoint_utils import GenerationCheckpoint

from .definitions import DatasetItemsLoader


@t.runtime_checkable
class DatasetWithGenerations(t.Protocol):
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


class CheckPointIteratorKwargs(t.TypedDict):
    """Optional Arguments for checkpoint iteration."""

    datasets: tuple[datasets.DATASET_NAMES, ...]
    model_types: tuple[lb_types.MODEL_TYPE, ...]
    temperatures: tuple[float, ...]
    langs: tuple[str, ...]
    splits: tuple[str, ...]
    chunks: tuple[str, ...]


@dataclasses.dataclass
class GenerationCheckpointLoader(DatasetItemsLoader):
    """Dataset loader for the checkpoint set."""

    lang: str
    split: str
    chunk: str
    temperature: float
    model_type: lb_types.MODEL_TYPE
    dt_cfg: DatasetWithGenerations

    @property
    def root_dir(self) -> Path:
        """Location to write to."""
        return self.dt_cfg.generation_checkpoint_root / self.lang / self.split / self.chunk / self.model_type

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

    def load_intermediate(self) -> GenerationCheckpoint | None:
        """Load intermediate checkpoint."""
        return GenerationCheckpoint.load_intermediate(location=self.root_dir, temperature=self.temperature)

    def load_final(self) -> GenerationCheckpoint | None:
        """Load final checkpoint."""
        return GenerationCheckpoint.load_final(location=self.root_dir, temperature=self.temperature)

    def exists(self) -> bool:
        """Return if ressources exist."""
        return self.root_dir.exists()

    def has_intermediate(self) -> bool:
        """Return if generation has started."""

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
        # Check if correct dataset is provided.
        if not isinstance(self.dt_cfg, DatasetWithGenerations):
            raise exc.DatasetTypeError(dataset=type(self.dt_cfg), protocol=DatasetWithGenerations)

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
        return cls(
            lang=lang,
            split=split,
            chunk=chunk,
            model_type=model_type,
            temperature=temperature,
            dt_cfg=datasets.get_config(dataset_name),
        )

    @classmethod
    def iter_items(cls, **kwargs: t.Unpack[CheckPointIteratorKwargs]) -> t.Iterator["GenerationCheckpointLoader"]:
        """Iterate over a set of items."""
        dataset_list = kwargs.get("datasets", ("stela", "child_realistic"))
        temperatures = kwargs.get("temperatures", settings.GENERATION_TEMPERATURES)
        for dt_name in dataset_list:
            dt_cfg: DatasetWithGenerations = datasets.get_config(dt_name)
            langs_list = kwargs.get("langs", dt_cfg.langs)
            split_list = kwargs.get("splits", dt_cfg.size_splits)
            chunk_list = kwargs.get("chunks", settings.TRAIN_CHUNKS)
            model_type_list = kwargs.get("model_types", settings.MODEL_TYPES)
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

                    for temp in temperatures:
                        for _model in model_type_list:
                            yield cls.load(
                                dataset_name=dt_cfg.dataset_name,
                                lang=_lang,
                                split=_split,
                                chunk=_chunk,
                                model_type=_model,
                                temperature=temp,
                            )


class GenTextIteratorKwargs(t.TypedDict):
    """Optional Arguments for checkpoint iteration."""

    model_type: lb_types.MODEL_TYPE


@dataclasses.dataclass
class GenerationItemsLoader(DatasetItemsLoader):
    """Dataset loader for the generated text architecture."""

    model_type: str
    estimation_type: str
    month: str
    temperature: float
    dt_cfg: ...

    @property
    def text_file(self) -> Path:
        """Path to the text file."""
        return (
            self.dt_cfg.generation_transcription_dir
            / self.dt_cfg.name
            / self.model_type
            / self.estimation_type
            / self.month
            / f"{self.temperature}.txt"
        )

    @classmethod
    def iter_items(cls, **kwargs) -> t.Iterable["GenerationItemsLoader"]:
        """Iterate over checkpoint items."""
        dataset_list = kwargs.get("datasets", ("stela", "child_realistic"))
        temperatures = kwargs.get("temperatures", settings.GENERATION_TEMPERATURES)
        for dt_name in dataset_list:
            dt_cfg: DatasetWithGenerations = datasets.get_config(dt_name)
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

                    for temp in temperatures:
                        yield cls.load(
                            dataset_name=dt_cfg.dataset_name,
                            lang=_lang,
                            split=_split,
                            chunk=_chunk,
                            temperature=temp,
                        )
