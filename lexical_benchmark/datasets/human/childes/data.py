import collections
import hashlib
import json
import string
import typing as t
from pathlib import Path

if t.TYPE_CHECKING:
    from lexical_benchmark.datasets.utils import DictionairyCleaner

import pandas as pd

from lexical_benchmark import settings, utils
from lexical_benchmark.datasets import utils as dataset_utils

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
        mdir = self.root_dir / lang / "meta" / speech_type
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

    def filter_words(self, lang_accent: str, speech_type: SPEECH_TYPE, cleaner: "DictionairyCleaner") -> None:
        """Filter words in dataset using a dictionairy."""
        for item in self.iter(lang_accent, speech_type):
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
            (meta_dir / "rejected" / f"{item.file_id}.txt").write_text("\n".join(rejected_lines))
            # Save source
            (meta_dir / "source" / f"{item.file_id}.txt").write_text("\n".join(source_text))

    def filter_words_txt(self, lang_accent: str, cleaner: "DictionairyCleaner") -> None:
        """Filter words in txt files."""

        def _line_clean(line: str) -> str:
            """Intern line clean."""
            clean, _ = cleaner(line)
            return clean

        for file in (self.root_dir / lang_accent / "txt").glob("*.json"):
            # Read & Clean
            as_obj = json.loads(file.read_bytes())
            clean_lines = [(label, _line_clean(line)) for label, line in as_obj]

            # Dump data back into dataset
            as_txt = json.dumps(clean_lines, indent=4, default=utils.default_json_encoder)
            file.with_suffix(".clean.json").write_text(as_txt)

    def restore_raw(self, lang_accent: str, speech_type: SPEECH_TYPE) -> None:
        """Restore word filtering."""
        for item in self.iter(lang_accent, speech_type):
            meta_dir = self.meta_location(item.lang_accent, item.speech_type)
            raw_backup_file = meta_dir / "source" / f"{item.file_id}.txt"
            if raw_backup_file.is_file():
                raw_text = raw_backup_file.read_text()
                item.file.write_text(raw_text)
                # Remove source
                raw_backup_file.unlink()

    def raw_word_frequency(self, lang_accent: str, speech_type: SPEECH_TYPE) -> pd.DataFrame:
        """Load or Compute raw word frequency for give speech type and language."""
        meta_dir = self.meta_location(lang_accent, speech_type)
        wf_file = meta_dir / "raw_word_frequency.csv"
        if wf_file.is_file():
            return pd.read_csv(wf_file, names=["word", "freq"], header=0)

        raw_files_list = []
        for item in self.iter(lang_accent, speech_type):
            raw_file = meta_dir / "source" / f"{item.file_id}.txt"
            if raw_file.is_file():
                raw_files_list.append(raw_file)

        df = dataset_utils.word_frequency_df(raw_files_list)
        df.to_csv(wf_file, index=False)
        return df

    def clean_word_frequency(self, lang_accent: str, speech_type: SPEECH_TYPE) -> pd.DataFrame:
        """Load or Compute clean word frequency for give speech type and language."""
        meta_dir = self.meta_location(lang_accent, speech_type)
        wf_file = meta_dir / "clean_word_frequency.csv"
        if wf_file.is_file():
            return pd.read_csv(wf_file, names=["word", "freq"], header=0)

        # Raw files
        files_list = [item.file for item in self.iter(lang_accent, speech_type)]

        df = dataset_utils.word_frequency_df(files_list)
        df.to_csv(wf_file, index=False)
        return df

    def rejected_word_frequency(self, lang_accent: str, speech_type: SPEECH_TYPE) -> pd.DataFrame:
        """Load or Compute rejected word frequency for give speech type and language."""
        meta_dir = self.meta_location(lang_accent, speech_type)
        wf_file = meta_dir / "rejected_word_frequency.csv"
        if wf_file.is_file():
            return pd.read_csv(wf_file, names=["word", "freq"], header=0)

        files_list = []
        for item in self.iter(lang_accent, speech_type):
            rejected_file = meta_dir / "rejected" / f"{item.file_id}.txt"
            if rejected_file.is_file():
                files_list.append(rejected_file)

        df = dataset_utils.word_frequency_df(files_list)
        df.to_csv(wf_file, index=False)
        return df

    def word_stats(self, lang_accent: str, speech_type: SPEECH_TYPE) -> dict[str, dict[str, float]]:
        """Compute stats on word retention."""
        # Raw words count
        raw_stats = self.raw_word_frequency(lang_accent, speech_type)

        # Validated words count
        clean_stats = self.clean_word_frequency(lang_accent, speech_type)

        # Rejected words count
        rejected_stats = self.rejected_word_frequency(lang_accent, speech_type)

        return {
            "raw": {
                "token_nb": raw_stats["freq"].sum(),
                "type_nb": len(raw_stats["word"]) - 1,
            },
            "clean": {
                "token_nb": clean_stats["freq"].sum(),
                "type_nb": len(clean_stats["word"]) - 1,
            },
            "bad": {
                "token_nb": rejected_stats["freq"].sum(),
                "type_nb": len(rejected_stats["word"]) - 1,
            },
        }


class CHILDESExtrasLexicon:
    """Loader for lexicon of extra words in childes tags."""

    EXTRAS_LABELS: t.ClassVar[tuple[str, ...]] = (
        "@o",  # Onomatopoeia
        "@p",  # Phonological Form
        "@b",  # Babbling
        "@wp",  # Word-play
        "@c",  # Child-Invented Form
        "@f",  # Family Form
        "@d",  # Dialect Words
        "@n",  # Neologisms
        "@i",  # Interjections
        "&-",  # Fillers
        "&~",  # Fillers
        "&+",  # Fragments
    )

    def __init__(self, raw_root_dir: Path = settings.PATH.raw_childes) -> None:
        self.raw_childes = RawCHILDESFiles(root_dir=raw_root_dir)
        self.words: set[str] = set()
        self.langs_speech: list[tuple[str, SPEECH_TYPE]] = []

    def add_words(self, word_list: list[str]) -> None:
        """Add words to dictionairy."""

        def clean(word: str) -> str:
            """Clean a word."""
            allowed_chars = string.ascii_lowercase + "' "
            word = word.replace("_", " ")
            return "".join(c.lower() for c in word if c.lower() in allowed_chars)

        clean_words = [clean(w) for w in word_list]
        self.words.update(clean_words)

    def add_lang(self, lang: str, speech_type: SPEECH_TYPE) -> None:
        """Add items from a language to the current dict."""
        # Add lang & speech_type to index
        self.langs_speech.append((lang, speech_type))

        # Iterate & extend word list from labels
        for file in self.raw_childes.iter_clean_meta(lang, speech_type):
            meta_dict = json.loads(file.read_bytes())
            words = []
            for label in self.EXTRAS_LABELS:
                words.extend(meta_dict.get(label, []))
            # Update global dict
            self.add_words(words)

    def current_fname(self) -> str:
        """Build a hash of wordlist specs to distinguish characteristics."""
        langs = "-".join(f"{a}_{b}" for a, b in self.langs_speech)
        source = f"{'-'.join(self.EXTRAS_LABELS)}||{langs}"
        return hashlib.md5(source.encode()).hexdigest()

    def cache_current(self) -> str:
        """Save current wordlist to cache."""
        location = settings.cache_dir()
        location = location / "childes_lexicon"
        location.mkdir(exist_ok=True, parents=True)
        fname = self.current_fname()

        as_dict = {
            "hash_id": fname,
            "languages": self.langs_speech,
            "childes_meta_tags": self.EXTRAS_LABELS,
            "word_count": len(self.words),
            "words": list(self.words),
        }
        as_json = json.dumps(as_dict, indent=4)
        (location / f"childes_extra_{fname}.json").write_text(as_json)
        return fname

    @classmethod
    def from_cache(cls, hash_id: str, raw_root_dir: Path = settings.PATH.raw_childes) -> "CHILDESExtrasLexicon":
        """Load dictionairy from cached file."""
        location = settings.cache_dir()
        cached_file = location / "childes_lexicon" / f"childes_extra_{hash_id}.json"
        if not cached_file.is_file():
            raise ValueError("Cached dict does not exist !!")

        as_dict = json.loads(cached_file.read_bytes())

        word_dict = cls(raw_root_dir=raw_root_dir)
        word_dict.words = set(as_dict.get("words", []))
        word_dict.langs_speech = as_dict.get("languages", [])

        if word_dict.current_fname() != hash_id:
            raise ValueError("Given hash does not match given dictionairy")

        return word_dict
