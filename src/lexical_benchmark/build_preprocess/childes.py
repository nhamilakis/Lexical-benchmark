import hashlib
import json
import logging
import string
import sys
import typing as t

import numpy as np
import pandas as pd
from rich.progress import track

from lexical_benchmark import datasets, exc, settings, utils
from lexical_benchmark.dataloaders import preprocess as preprocess_dataloaders
from lexical_benchmark.text_lib import parsing

if t.TYPE_CHECKING:
    from pathlib import Path


def parse_childes_age(age: str) -> float:
    """Parse string based age and convert it into months.

    The age of the childs is formated in the folowing way :

                "Y;MM.MM"

    where Y is the number of years, MM is the number of months.
    The year is an int and the months are formatted as float (to include days).

    Returns
    -------
        float: as the number of months

    """
    age = str(age)
    if ";" not in age:
        try:
            return float(age)
        except ValueError:
            return np.nan

    year, _, month = age.partition(";")
    try:
        return (int(year) * 12) + float(month)
    except ValueError:
        return np.nan


class CHILDESPreparation:
    """Class used to pre-format the CHILDES dataset.

    Output Preprocess Dataset Structure:
    preprocess_dir/
        LANG_ACCENT/
            child_speech/
                *.raw
            adult_speech/
                *.raw
            dialogs/
                *.raw.json

    """

    def __init__(self) -> None:
        self.cfg: datasets.CHILDESDatasetConfig = datasets.get_config("childes")
        self.dataset: dict[str, list[tuple[str, Path]]] = {}
        self.id_list: dict[str, list[tuple[str, ...]]] = {}
        self.parsing_verbosity("INFO")

    @staticmethod
    def parsing_verbosity(level: str) -> None:
        """Set custom verbosity for the parsing module."""
        parsing_logger = logging.getLogger("lexical_benchmark.text_lib.parsing.cha.extract")
        parsing_logger.setLevel(level)

    def load_dir(self, lang_code: str) -> None:
        """Load .cha files by Language."""
        all_items = []
        root_dir = self.cfg.original_root / lang_code
        id_list = []
        for item in root_dir.rglob("*.cha"):
            file_id = str(item.relative_to(root_dir).parent).replace("/", "_")
            id_list.append(tuple(file_id.split("_")))
            file_id = f"{file_id}_{item.stem}"
            all_items.append((file_id, item))
        self.dataset[lang_code] = all_items
        self.id_list[lang_code] = id_list

    def write(self, lang_code: str) -> pd.DataFrame | None:
        """Parse id list as dataframe."""
        id_list = self.id_list.get(lang_code, [])
        if len(id_list) == 0:
            return None

    def build_preprocess(self, *, show_progress: bool = False) -> None:
        """Export dataset to directory."""
        meta_dir = self.cfg.meta_dir

        for lang_code, filelist in self.dataset.items():
            if lang_code not in self.cfg.all_accents:
                raise exc.UnknownDatasetLangError(lang=lang_code)

            root_dir = self.cfg.preprocessed_root / lang_code
            root_dir.mkdir(parents=True, exist_ok=True)
            metadata = []

            for name, file in track(
                filelist, description=f"Processing {lang_code} .cha files...", disable=not show_progress
            ):
                try:
                    data_typed = parsing.cha.extract(file, name)  # type: ignore[call-arg]

                    (root_dir / "child" / f"{name}.raw").safe_write_text("\n".join(data_typed.child_speech))
                    (root_dir / "adult" / f"{name}.raw").safe_write_text("\n".join(data_typed.adult_speech))
                    metadata.append(data_typed.csv_entry())

                    data_dialog = parsing.cha.extract_with_tags(file, name)  # type: ignore[call-arg]
                    as_json = json.dumps(data_dialog.speech, indent=4, default=utils.default_json_encoder)
                    (root_dir / "dialogs" / f"{name}.raw.json").safe_write_text(as_json)

                except UnicodeDecodeError:
                    print(f"Failed to process {file}...", file=sys.stderr)
                    raise

            df = pd.DataFrame(metadata, columns=["file_id", "lang", "child_gender", "child_age"])
            meta_dir.mkdir(exist_ok=True, parents=True)
            df.to_csv(meta_dir / f"child_metadata_{lang_code}.csv", index=False)


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

    def __init__(self) -> None:
        self.dataset_cfg: datasets.CHILDESDatasetConfig = datasets.get_config("childes")
        self.words: set[str] = set()
        self.langs_speech: list[tuple[str, datasets.CHILDES_SPEECH_TYPES]] = []

    def add_words(self, word_list: list[str]) -> None:
        """Add words to dictionairy."""

        def clean(word: str) -> str:
            """Clean a word."""
            allowed_chars = string.ascii_lowercase + "' "
            word = word.replace("_", " ")
            return "".join(c.lower() for c in word if c.lower() in allowed_chars)

        clean_words = [clean(w) for w in word_list]
        self.words.update(clean_words)

    def add_lang(self, lang_accent: str, speech_type: datasets.CHILDES_SPEECH_TYPES) -> None:
        """Add items from a language to the current dict."""
        if lang_accent not in self.dataset_cfg.all_accents:
            raise exc.UnknownDatasetLangError(lang_accent)

        self.langs_speech.append((lang_accent, speech_type))
        # self.childes.iter_accent(accent=lang_accent):

        cfg_kwargs = {
            "lang_accents": (lang_accent,),
            "speech_types": (speech_type,),
        }

        for item in preprocess_dataloaders.CHILDESPreprocessedItems.iter_items(**cfg_kwargs):
            meta_dict = json.loads(item.cleanup_meta.read_bytes())
            words = []
            for label in self.EXTRAS_LABELS:
                words.extend(meta_dict.get(label, []))
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
    def from_cache(cls, hash_id: str) -> "CHILDESExtrasLexicon":
        """Load dictionairy from cached file."""
        location = settings.cache_dir()
        cached_file = location / "childes_lexicon" / f"childes_extra_{hash_id}.json"
        if not cached_file.is_file():
            raise ValueError("Cached dict does not exist !!")

        as_dict = json.loads(cached_file.read_bytes())

        word_dict = cls()
        word_dict.words = set(as_dict.get("words", []))
        word_dict.langs_speech = as_dict.get("languages", [])

        if word_dict.current_fname() != hash_id:
            raise ValueError("Given hash does not match given dictionairy")

        return word_dict
