import itertools
import json
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from lexical_benchmark import datasets

from .definitions import DatasetItemsLoader

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
class CHILDESTXTAccessor:
    """Accessor for txt aggregates."""

    lang: str
    dt_cfg: datasets.CHILDESDatasetConfig = field(default_factory=lambda: datasets.get_config("childes"))

    def load_text(self, speech_type: datasets.CHILDES_SPEECH_TYPES) -> list[str]:
        """Load text file."""
        txt_file = self.dt_cfg.text_dir / speech_type / f"{self.lang}.txt"
        return txt_file.safe_readlines()

    def child_by_age(self) -> list[tuple[int, list[str]]]:
        """Load child-speech by age."""
        # TODO: add this if necessairy for child-model comparison
        raise NotImplementedError("Requires implementation")


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
