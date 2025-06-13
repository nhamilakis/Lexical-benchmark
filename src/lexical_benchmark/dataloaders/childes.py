import itertools
import json
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import typing_extensions as t_extra

from lexical_benchmark import datasets, lb_types, settings, text_lib, utils
from lexical_benchmark.text_lib import tokenization

from .definitions import DatasetItemsLoader, DatasetTrainArgLoader

DialogFormatType = list[tuple[str, str]]


class DatasetWithDialogs(t.Protocol):
    """Dataset config with support for dialog data."""

    @property
    def by_dialogs(self) -> Path:
        """Dialog path."""

    @property
    def all_accents(self) -> tuple[str, ...]:
        """List of accents."""

    @property
    def langs(self) -> tuple[str, ...]:
        """List of languages."""

    @property
    def preprocessed_root(self) -> Path:
        """Location to preprocessed files."""


def check_speech(speaker: str, speech_type: datasets.CHILDES_SPEECH_TYPES) -> bool:
    """Check if speaker is in speech_type."""
    match speaker:
        case "CHI":
            return speech_type == "child"
        case _:
            return speech_type == "adult"


@dataclass
class CHILDESTrainItem(DatasetTrainArgLoader):
    """Train arguments for CHILDES."""

    speech_type: datasets.CHILDES_SPEECH_TYPES
    data_item: "CHILDESTXTAccessor"
    model_type: lb_types.MODEL_TYPE
    resume: bool = True
    override: bool = False
    resume_id: str | None = None
    batch_size: int = 32

    @property
    @t_extra.override
    def item_id(self) -> str:
        """Unique identifier for item."""
        return f"childes_{self.data_item.lang}_{self.speech_type}"

    @property
    @t_extra.override
    def job_name(self) -> str:
        """Unique train item identifier."""
        return f"{self.dataset_name}_{self.model_type}_{self.item_id}"

    @property
    @t_extra.override
    def dataset_name(self) -> str:
        """Name of the dataset."""
        return self.data_item.dt_cfg.dataset_name

    @property
    @t_extra.override
    def model_root_dir(self) -> Path:
        """Root directory."""
        return settings.PATH.model_root / self.dataset_name / self.speech_type / self.model_type

    @property
    @t_extra.override
    def generation_root_dir(self) -> Path:
        """Root directory."""
        return settings.PATH.generate_root / self.dataset_name / self.speech_type / self.model_type

    def __load__format_text(self) -> None:
        """Load & format childes text."""
        text = self.data_item.load_text(self.speech_type)
        dev_txt, train_txt = text_lib.split_lines_by_tokens(
            text,
            train_ratio=(100 - 8.5) / 100,
            random_seed=settings.RANDOM_SEED,
        )
        # Write as split
        self.data_item.text_dev_file(self.speech_type).safe_write_text("\n".join(dev_txt))
        self.data_item.text_train_file(self.speech_type).safe_write_text("\n".join(train_txt))

        # Write tokenized versions
        self.data_item.text_dev_tokenized_file(self.speech_type).safe_write_text(
            "\n".join([tokenization.hf_line_format(line) for line in dev_txt])
        )
        self.data_item.text_train_tokenized_file(self.speech_type).safe_write_text(
            "\n".join([tokenization.hf_line_format(line) for line in train_txt])
        )

    @t_extra.override
    def train_txt(self) -> Path:
        """Load train data."""
        text_file = self.data_item.text_train_tokenized_file(self.speech_type)
        if not text_file.is_file():
            self.__load__format_text()
        return text_file

    @t_extra.override
    def dev_txt(self) -> Path:
        """Load dev data."""
        text_file = self.data_item.text_dev_tokenized_file(self.speech_type)
        if not text_file.is_file():
            self.__load__format_text()
        return text_file

    def to_dict(self) -> dict:
        """Convert item into a dictionairy."""
        last_checkpoint = self.last_checkpoint()
        return {
            "model_type": self.model_type,
            "lang": self.data_item.lang,
            "dataset_name": self.dataset_name,
            "speech_type": self.speech_type,
            "completed_training": self.completed_training,
            "generation_root": str(self.generation_root_dir) if self.generation_root_dir else self.generation_root_dir,
            "model_root": str(self.model_root_dir) if self.model_root_dir else self.model_root_dir,
            "last_checkpoint": str(last_checkpoint) if last_checkpoint else last_checkpoint,
            "resume": self.resume,
            "override": self.override,
            "resume_id": self.resume_id,
            "batch_size": self.batch_size,
        }

    @classmethod
    @t_extra.override
    def from_dict(cls, cfg_args: dict) -> t_extra.Self:
        """Load train args from a dictionairy."""
        return cls(
            data_item=CHILDESTXTAccessor(
                lang=cfg_args["lang"],
            ),
            model_type=cfg_args["model_type"],
            speech_type=cfg_args["speech_type"],
            resume=utils.str_to_bool(cfg_args["resume"]),
            override=utils.str_to_bool(cfg_args["override"]),
            resume_id=cfg_args["resume_id"],
            batch_size=cfg_args["batch_size"],
        )


@dataclass
class CHILDESTXTAccessor:
    """Accessor for txt aggregates."""

    lang: str
    dt_cfg: datasets.CHILDESDatasetConfig = field(default_factory=lambda: datasets.get_config("childes"))

    def text_file(self, speech_type: datasets.CHILDES_SPEECH_TYPES) -> Path:
        """Location to text file."""
        return self.dt_cfg.text_dir / speech_type / f"{self.lang}.txt"

    def text_dev_file(self, speech_type: datasets.CHILDES_SPEECH_TYPES) -> Path:
        """Location to text file."""
        return self.text_file(speech_type).with_suffix(".dev.txt")

    def text_train_file(self, speech_type: datasets.CHILDES_SPEECH_TYPES) -> Path:
        """Location to text file."""
        return self.text_file(speech_type).with_suffix(".train.txt")

    def text_dev_tokenized_file(self, speech_type: datasets.CHILDES_SPEECH_TYPES) -> Path:
        """Location to text file."""
        return self.text_file(speech_type).with_suffix(".tokenized.dev.txt")

    def text_train_tokenized_file(self, speech_type: datasets.CHILDES_SPEECH_TYPES) -> Path:
        """Location to text file."""
        return self.text_file(speech_type).with_suffix(".tokenized.train.txt")

    def load_text(self, speech_type: datasets.CHILDES_SPEECH_TYPES) -> list[str]:
        """Load text file."""
        return self.text_file(speech_type).safe_readlines()

    def child_by_age(self) -> list[tuple[int, list[str]]]:
        """Load child-speech by age."""
        # TODO: add this if necessairy for child-model comparison
        raise NotImplementedError("Requires implementation")

    def train_args(
        self,
        *,
        model_type: lb_types.MODEL_TYPE,
        speech_type: datasets.CHILDES_SPEECH_TYPES,
        resume: bool = True,
        override: bool = False,
        resume_id: str | None = None,
    ) -> "CHILDESTrainItem":
        """Load train arguments."""
        return CHILDESTrainItem(
            data_item=self,
            model_type=model_type,
            speech_type=speech_type,
            resume=resume,
            override=override,
            resume_id=resume_id,
        )


@dataclass
class CHILDESTextLoader(DatasetItemsLoader):
    """Loader for clean txt items in the CHILDES dataset, from dialogs."""

    lang_accent: str
    item_id: str
    speech_type: datasets.CHILDES_SPEECH_TYPES
    dt_cfg: DatasetWithDialogs = field(default_factory=lambda: datasets.get_config("childes"))

    @classmethod
    def load(
        cls, *, lang_accent: str, item_id: str, speech_type: datasets.CHILDES_SPEECH_TYPES
    ) -> "DatasetItemsLoader":
        """Load item directly."""
        return cls(lang_accent=lang_accent, item_id=item_id, speech_type=speech_type)

    @property
    def source_file(self) -> Path:
        """Path to the text of the current item."""
        return self.dt_cfg.by_dialogs / self.lang_accent / f"{self.item_id}.json"

    def load_speech(self) -> t.Iterable[str]:
        """Load current speech type."""
        return iter(text for [speaker, text] in self.source_file.read_json() if check_speech(speaker, self.speech_type))

    @classmethod
    def iter_items(cls, **kwargs) -> t.Iterable["CHILDESTextLoader"]:
        """Iterate over preprocessed items."""
        cfg: datasets.CHILDESDatasetConfig = datasets.get_config("childes")
        langs_list = kwargs.get("langs", cfg.langs)
        speech_types = kwargs.get("speech_types", cfg.SPEECH_TYPES)

        if "lang_accents" not in kwargs:
            lang_accents = [cfg.LANG_ACCENT.get(lang, ()) for lang in langs_list]
            lang_accents = tuple(itertools.chain(*lang_accents))
        else:
            lang_accents = kwargs.get("lang_accents")

        for lg_accent in lang_accents:
            if lg_accent not in cfg.all_accents:
                continue

            for item_parts in cfg.id_list(lang_accent=lg_accent):
                item_id = "_".join(item_parts)

                for spt in speech_types:
                    if spt not in cfg.SPEECH_TYPES:
                        continue

                    # Return invidivual items
                    yield cls.load(
                        lang_accent=lg_accent,
                        speech_type=spt,
                        item_id=item_id,
                    )


@dataclass
class CHILDESDialogLoader(DatasetItemsLoader):
    """Loader for Dialog formatted data."""

    lang_accent: str
    item_id: str
    dt_cfg: DatasetWithDialogs = field(default_factory=lambda: datasets.get_config("childes"))

    @property
    def root_dir(self) -> Path:
        """Current item root dir."""
        return self.dt_cfg.by_dialogs / self.lang_accent

    @property
    def dialog_file(self) -> Path:
        """Path to dialog file (JSON format)."""
        return self.root_dir / f"{self.item_id}.json"

    @classmethod
    def iter_items(cls, **kwargs) -> t.Iterable["CHILDESDialogLoader"]:
        """Iterator over dialog items."""
        cfg: datasets.CHILDESDatasetConfig = datasets.get_config("childes")
        langs_list = kwargs.get("langs", cfg.langs)
        if "lang_accents" not in kwargs:
            lang_accents = [cfg.LANG_ACCENT.get(lang, ()) for lang in langs_list]
            lang_accents = tuple(itertools.chain(*lang_accents))
        else:
            lang_accents = kwargs.get("lang_accents")

        for lg_accent in lang_accents:
            if lg_accent not in cfg.all_accents:
                continue

            for item_parts in cfg.id_list(lang_accent=lg_accent):
                item_id = "_".join(item_parts)

                # Return invidivual items
                yield cls(
                    lang_accent=lg_accent,
                    item_id=item_id,
                )

    @classmethod
    def item_dialogs(cls, **kwargs) -> t.Iterable[tuple["CHILDESDialogLoader", DialogFormatType]]:
        """Iterator that load dialogs."""
        for item in cls.iter_items(**kwargs):
            with item.dialog_file.open() as fh:
                data = json.load(fh)
            yield (item, data)


@dataclass
class TurnTakingCSVData:
    """Struct containing turn-taking format."""

    adult_label: str
    adult: t.Literal["<EMPTY>"] | str  # noqa: PYI051
    child: t.Literal["<EMPTY>"] | str  # noqa: PYI051
    COLUMNS: t.ClassVar[tuple[str, ...]] = ("label", "adult_speech", "child_speech")

    def as_row(self) -> tuple[str, str, str]:
        """Row used to build turn-take data as csv."""
        return self.adult_label, self.adult, self.child


@dataclass
class TurnTakeDataLoader:
    """Representation of turn-taking format."""

    lang_accent: str
    file_id: str

    @property
    def root_dir(self) -> Path:
        """Location of the current root dir."""
        return self.dt_cfg.by_turn / self.lang_accent

    @property
    def csv_path(self) -> Path:
        """Path to the current set data."""
        return self.root_dir / f"{self.file_id}.csv"

    def __post_init__(self) -> None:
        self.dt_cfg: datasets.CHILDESDatasetConfig = datasets.get_config("childes")

    if pl:

        def load_df(self) -> "pl.DataFrame":
            """Load as dataframe."""
            return pl.read_csv(self.csv_path)
    else:

        def load_df(self) -> pl.DataFrame:
            """Load as dataframe."""
            return pl.read_csv(self.csv_path)

    def load(self) -> list[TurnTakingCSVData]:
        """Load data from CSV."""
        df = self.load_df()

        # Get field names from the dataclass
        expected_columns = set(TurnTakingCSVData.COLUMNS)
        found_columns = set(df.columns)
        if not expected_columns.issubset(found_columns):
            missing = expected_columns - found_columns
            raise ValueError(f"Missing columns in CSV: {missing}")

        # Only select columns that are in the dataclass
        df_filtered = df.select([col for col in df.columns if col in expected_columns])
        return [TurnTakingCSVData(**row) for row in df_filtered.to_dicts()]

    def save(self, rows: list[TurnTakingCSVData]) -> None:
        """Save a list of rows into the csv form."""
        df = pl.DataFrame(rows, schema=TurnTakingCSVData.COLUMNS)
        df.write_csv(self.csv_path, include_header=True)

    @classmethod
    def iter_items(cls, **kwargs) -> t.Iterable["TurnTakeDataLoader"]:
        """Iterate over all by_turn items."""
        cfg: datasets.CHILDESDatasetConfig = datasets.get_config("childes")
        langs_list = kwargs.get("langs", cfg.langs)
        if "lang_accents" not in kwargs:
            lang_accents = [cfg.LANG_ACCENT.get(lang, ()) for lang in langs_list]
            lang_accents = tuple(itertools.chain(*lang_accents))
        else:
            lang_accents = kwargs.get("lang_accents")

        for lg_accent in lang_accents:
            if lg_accent not in cfg.all_accents:
                continue

            for item_parts in cfg.id_list(lang_accent=lg_accent):
                item_id = "_".join(item_parts)

                # Return invidivual items
                yield cls(
                    lang_accent=lg_accent,
                    item_id=item_id,
                )
