import collections
import json
import typing as t
from pathlib import Path

from lexical_benchmark import settings
from lexical_benchmark.datasets import utils

SPEECH_TYPE = t.Literal["adult", "child"]


class CHAItem(t.NamedTuple):
    """Item containing information for CHA file."""

    lang_accent: str
    file_id: str
    file: Path


class TXTItem(t.NamedTuple):
    """Item containing information for CHA file."""

    lang_accent: str
    file_id: str
    file: Path
    speech_type: SPEECH_TYPE


class SourceCHILDESFiles:
    """Navigation into Files from Childes Source Version.

    The source version of CHILDES is the one that is in the raw format as it was downloaded from the
    talkbank.org website.
    """

    @property
    def langs(self) -> tuple[str, ...]:
        """List of languages (Accents) in the dataset."""
        return settings.CHILDES.ACCENTS

    def __init__(self, root_dir: Path = settings.PATH.source_childes) -> None:
        self.root_dir = root_dir

        if not root_dir.is_dir():
            raise ValueError(f"Cannot load non-existent directory: {root_dir}")

    def iter(self, lang_accent: str) -> t.Iterable[CHAItem]:
        """Iterate over source files."""
        if lang_accent not in self.langs:
            raise KeyError(f"{lang_accent}: Not found !!")

        for cha_file in (self.root_dir / lang_accent).rglob("*.cha"):
            file_id = str(cha_file.relative_to(self.root_dir / lang_accent))
            file_id = file_id.replace("/", "_")

            yield CHAItem(
                lang_accent,
                file_id,
                cha_file,
            )

    def iter_all(self) -> t.Iterable[CHAItem]:
        """Iterate over source files."""
        for lang_accent in self.langs:
            yield from self.iter(lang_accent)


class RawCHILDESFiles:
    """Navigation into post refacter raw CHILDES dataset."""

    @property
    def langs(self) -> tuple[str, ...]:
        """List of languages (Accents) in the dataset."""
        return settings.CHILDES.ACCENTS

    def __init__(self, root_dir: Path = settings.PATH.raw_childes) -> None:
        self.root_dir = root_dir

        if not root_dir.is_dir():
            raise ValueError(f"Cannot load non-existent directory: {root_dir}")

    def iter(self, lang_accent: str, speech_type: SPEECH_TYPE) -> t.Iterable[TXTItem]:
        """Iterate over source files."""
        if lang_accent not in self.langs:
            raise KeyError(f"{lang_accent}: Not found !!")

        for file in (self.root_dir / lang_accent / speech_type).glob("*.raw"):
            yield TXTItem(
                lang_accent=lang_accent,
                file_id=file.stem,
                file=file,
                speech_type=speech_type,
            )

    def iter_all(self, speech_type: SPEECH_TYPE) -> t.Iterable[TXTItem]:
        """Iterate over all items."""
        for lang_accent in self.langs:
            yield from self.iter(lang_accent, speech_type=speech_type)

    def iter_clean_meta(self, lang_accent: str, speech_type: SPEECH_TYPE) -> t.Iterable[Path]:
        """Iterate over all post-cleanup metadata file."""
        if lang_accent not in self.langs:
            raise KeyError(f"{lang_accent}: Not found !!")

        yield from (self.root_dir / lang_accent / speech_type).glob("*.meta.json")

    def load_clean_meta(self, lang: str, speech_type: SPEECH_TYPE) -> dict[str, int | list[str]]:
        """Function that allows to load all metadata from RAW Dataset."""
        # Load all metadata
        dict_list = [json.loads(f.read_bytes()) for f in self.iter_clean_meta(lang, speech_type)]

        # Load all keys
        keys = []
        for d in dict_list:
            keys.extend(list(d.keys()))
        keys_set = set(keys)

        # Load items of each key
        data: dict[str, list[str]] = collections.defaultdict(list)
        nb_data: dict[str, int] = collections.defaultdict(lambda: 0)

        for key in keys_set:
            for d in dict_list:
                item = d.get(key, [])
                if isinstance(item, int):
                    nb_data[key] += item
                else:
                    data[key].extend(item)

        return {**data, **nb_data}


class CleanCHILDESFiles:
    """Navigation into post-cleanup CHILDES dataset."""

    @property
    def age_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return list of age ranges ((MIN,MAX), ...)."""
        return settings.CHILDES.AGE_RANGES

    @property
    def langs(self) -> tuple[str, ...]:
        """List of languages (Accents) in the dataset."""
        return settings.CHILDES.ACCENTS

    def meta_location(self, lang: str, speech_type: SPEECH_TYPE) -> Path:
        """Return location for metadata."""
        mdir = self.root_dir / lang / "meta"
        mdir.mkdir(exist_ok=True, parents=True)
        return mdir

    def __init__(self, root_dir: Path = settings.PATH.clean_childes) -> None:
        self.root_dir = root_dir

        if not root_dir.is_dir():
            raise ValueError(f"Cannot load non-existent directory: {root_dir}")

    def get_child(self, lang: str, file_id: str) -> Path:
        """Get specific child speech file."""
        return self.root_dir / lang / "child" / f"{file_id}.txt"

    def get_adult(self, lang: str, file_id: str) -> Path:
        """Get specific adult speech file."""
        return self.root_dir / lang / "adult" / f"{file_id}.txt"

    def iter_by_age(self, lang_accent: str) -> t.Iterable[tuple[str, TXTItem]]:
        """Iterate over files by age groups."""
        if lang_accent not in self.langs:
            raise KeyError(f"{lang_accent}: Not found !!")

        location = self.root_dir / lang_accent
        for min_age, max_age in self.age_ranges:
            min_max = f"{min_age}_{max_age}"
            for file in (location / min_max).glob("*.txt"):
                yield (
                    min_max,
                    TXTItem(
                        lang_accent=lang_accent,
                        file_id=file.stem,
                        file=file,
                        speech_type="child",
                    ),
                )

    def iter(self, lang_accent: str, speech_type: SPEECH_TYPE) -> t.Iterable[TXTItem]:
        """Iterate over files by age groups."""
        if lang_accent not in self.langs:
            raise KeyError(f"{lang_accent}: Not found !!")

        location = self.root_dir / lang_accent / speech_type
        for file in location.glob("*.txt"):
            yield TXTItem(
                lang_accent=lang_accent,
                file_id=file.stem,
                file=file,
                speech_type=speech_type,
            )

    def iter_all(self, speech_type: SPEECH_TYPE) -> t.Iterable[TXTItem]:
        """Iterate over all files."""
        for lang_accent in self.langs:
            yield from self.iter(lang_accent=lang_accent, speech_type=speech_type)

    def iter_test(self) -> t.Iterable[Path]:
        """Iterate over the test files: Only used during dev."""
        yield from (self.root_dir / "test").glob("*.txt")

    def filter_words(self, speech_type: SPEECH_TYPE, cleaner: utils.DictionairyCleaner) -> None:
        """Filter words in dataset using a dictionairy."""
        for item in self.iter_all(speech_type):
            meta_dir = self.meta_location(item.lang_accent, item.speech_type)
            (meta_dir / "rejected").mkdir(exist_ok=True, parents=True)
            (meta_dir / "source").mkdir(exist_ok=True, parents=True)
            accepted_lines = []
            rejected_lines = []

            source_text = item.file.read_text().splitlines()
            for line in source_text:
                accepted, rejected = cleaner(line)
                accepted_lines.append(accepted)
                rejected_lines.append(rejected)

            # Write clean text back into file
            item.file.write_text("\n".join(accepted_lines))

            # Save Rejected
            (meta_dir / "rejected" / f"{item.file_id}").write_text("\n".join(rejected_lines))
            # Save source
            (meta_dir / "source" / f"{item.file_id}").write_text("\n".join(rejected_lines))

    def raw_word_frequency(self, lang: str, speech_type: SPEECH_TYPE):
        """Load or Compute raw word frequency for give speech type and language."""
        # TODO(@nhamilakis): implement
        pass

    def clean_word_frequency(self, lang: str, speech_type: SPEECH_TYPE):
        """Load or Compute clean word frequency for give speech type and language."""
        # TODO(@nhamilakis): implement
        pass

    def rejected_word_frequency(self, lang: str, speech_type: SPEECH_TYPE):
        """Load or Compute rejected word frequency for give speech type and language."""
        # TODO(@nhamilakis): implement
        pass

    def word_stats(self, lang: str, speech_type: SPEECH_TYPE):
        """Compute stats on word retention."""
        # TODO(@nhamilakis): implement
        all_count = ...  # All words
        good_count = ...  # clean word count
        bad_count = ...  # Rejected word count

        return {
            "all": all_count,
            "good": good_count,
            "bad": bad_count,
            "bad_percent": (bad_count / all_count) * 100,
            "good_percent": (good_count / all_count) * 100,
        }
