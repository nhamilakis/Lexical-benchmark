import hashlib
import json
import string
import sys
import typing as t
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rich.progress import track

from lexical_benchmark import settings, utils
from lexical_benchmark.datasets.utils import parsing

from .data import SPEECH_TYPES, CHILDESDataset


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

    Output Dataset Structure:
        EN_UK/
            metadata.csv
            child_speech/
                *.raw
            adult_speech/
                *.raw
        EN_NA/
            metadata.csv
            child_speech/
                *.raw
            adult_speech/
                *.raw
    """

    def __init__(self) -> None:
        self.dataset: dict[str, list[tuple[str, Path]]] = {}

    def load_dir(self, root_dir: Path, lang_code: str) -> None:
        """Load .cha files by Language."""
        all_items = []
        for item in root_dir.rglob("*.cha"):
            file_id = str(item.relative_to(root_dir).parent).replace("/", "_")
            file_id = f"{file_id}_{item.stem}"
            all_items.append((file_id, item))
        self.dataset[lang_code] = all_items

    def export(self, location: Path, *, meta_dir: Path | None = None, show_progress: bool = False) -> None:
        """Export dataset to directory."""
        meta_dir = meta_dir if meta_dir else location
        for lang, filelist in self.dataset.items():
            lang_loc = location / lang
            lang_loc.mkdir(parents=True, exist_ok=True)
            metadata = []

            for name, file in track(
                filelist, description=f"Processing {lang} .cha files...", disable=not show_progress
            ):
                try:
                    data = parsing.cha.extract(file, name)  # type: ignore[call-arg]
                    (lang_loc / "child" / f"{name}.raw").safe_write_text("\n".join(data.child_speech))
                    (lang_loc / "adult" / f"{name}.raw").safe_write_text("\n".join(data.adult_speech))
                    metadata.append(data.csv_entry())
                except UnicodeDecodeError:
                    print(f"Failed to process {file}...", file=sys.stderr)
                    raise
            df = pd.DataFrame(metadata, columns=["file_id", "lang", "child_gender", "child_age"])
            meta_dir.mkdir(exist_ok=True, parents=True)
            df.to_csv(meta_dir / f"metadata_{lang}.csv", index=False)

    def export_turn_taking(self, location: Path) -> None:
        """Export dataset in the turntaking format to given directory."""
        for lang, filelist in self.dataset.items():
            lang_loc = location / lang
            lang_loc.mkdir(parents=True, exist_ok=True)
            for name, file in filelist:
                try:
                    data = parsing.cha.extract_with_tags(file, name)  # type: ignore[call-arg]
                    as_json = json.dumps(data.speech, indent=4, default=utils.default_json_encoder)
                    (lang_loc / "txt" / f"{name}.json").safe_write_text(as_json)

                except UnicodeDecodeError:
                    print(f"Failed to process {file}...", file=sys.stderr)
                    raise


class OrganizeByAge:
    """Organize child-speech by age."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def make_age_splits(self, lang_code: str) -> pd.DataFrame:
        """Create a split of metadata.csv into age groups."""
        data_dir = self.root_dir / lang_code
        metadata_df = pd.read_csv(data_dir / "metadata.csv", sep=",")
        metadata_df["child_age(float)"] = metadata_df["child_age"].apply(parse_childes_age)

        # Group by age groups
        metadata_df["age_group"] = pd.cut(
            metadata_df["child_age(float)"],
            bins=range(settings.CHILDES.MAX_AGE),
            labels=[f"{m}_{n}" for m, n in settings.CHILDES.AGE_RANGES],
        )

        # Remove ages outside of range
        return metadata_df[~metadata_df["age_group"].isna()]

    def build_splits(self, lang_code: str, metadata: pd.DataFrame) -> None:
        """Build folders separating children by age."""
        data_dir = self.root_dir / lang_code
        target = data_dir / "child_by_age"
        target.mkdir(exist_ok=True, parents=True)

        for min_age, max_age in settings.CHILDES.AGE_RANGES:
            curr_range = f"{min_age}_{max_age}"
            files_list = list(metadata[metadata["age_group"] == curr_range]["file_id"])
            # Make directory
            (target / curr_range).mkdir(exist_ok=True, parents=True)

            # Create symlink for all files
            for file_id in files_list:
                (target / curr_range / f"{file_id}.txt").symlink_to(data_dir / "child" / f"{file_id}.txt")


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

    def __init__(self, childes_dataset: CHILDESDataset | None) -> None:
        if childes_dataset is None:
            self.childes_dataset = CHILDESDataset()
        else:
            self.childes = childes_dataset
        self.words: set[str] = set()
        self.langs_speech: list[tuple[str, SPEECH_TYPES]] = []

    def add_words(self, word_list: list[str]) -> None:
        """Add words to dictionairy."""

        def clean(word: str) -> str:
            """Clean a word."""
            allowed_chars = string.ascii_lowercase + "' "
            word = word.replace("_", " ")
            return "".join(c.lower() for c in word if c.lower() in allowed_chars)

        clean_words = [clean(w) for w in word_list]
        self.words.update(clean_words)

    def add_lang(self, lang_accent: str, speech_type: "SPEECH_TYPES") -> None:
        """Add items from a language to the current dict."""
        # Add lang & speech_type to index
        self.langs_speech.append((lang_accent, speech_type))

        # Iterate & extend word list from labels
        for item in self.childes.iter_accent(accent=lang_accent):
            src_item = item.preprocess_item(speech_type)
            meta_dict = json.loads(src_item.meta.read_bytes())
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
    def from_cache(cls, hash_id: str, childes_dataset: CHILDESDataset | None) -> "CHILDESExtrasLexicon":
        """Load dictionairy from cached file."""
        location = settings.cache_dir()
        cached_file = location / "childes_lexicon" / f"childes_extra_{hash_id}.json"
        if not cached_file.is_file():
            raise ValueError("Cached dict does not exist !!")

        as_dict = json.loads(cached_file.read_bytes())

        word_dict = cls(childes_dataset=childes_dataset)
        word_dict.words = set(as_dict.get("words", []))
        word_dict.langs_speech = as_dict.get("languages", [])

        if word_dict.current_fname() != hash_id:
            raise ValueError("Given hash does not match given dictionairy")

        return word_dict


@dataclass
class TurnTakeData:
    """Representation of turn-taking format."""

    adult_label: str
    adult: t.Literal["<EMPTY>"] | str  # noqa: PYI051
    child: t.Literal["<EMPTY>"] | str  # noqa: PYI051
    file_id: str = ""  # FileID is optional
    COLUMNS: t.ClassVar[tuple[str, ...]] = ("label", "adult_speech", "child_speech")

    def row(self) -> tuple[str, str, str]:
        """Row used to build turn-take data as csv."""
        return self.adult_label, self.adult, self.child


class TurnTakingBuilder:
    """Builder class for turn-taking sub-dataset."""

    @property
    def langs(self) -> tuple[str, ...]:
        """Languages Included in chiles."""
        return settings.CHILDES.ACCENTS

    def iter(self, lang: str) -> t.Iterable[Path]:
        """Iterator over files."""
        root = self.root_dir / lang / "txt"
        if not root.is_dir():
            raise ValueError(f"Lang {lang} not found in dataset")

        yield from root.glob("*.clean.json")

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    @staticmethod
    def merge_consecutive_speakers(dialog: list[list[str] | tuple[str, str]]) -> list[tuple[str, str]]:
        """Merge consecutive speakers in a dialog list."""
        merged_dialog: list[tuple[str, str]] = []

        if not dialog:  # If the input list is empty, return an empty list
            return merged_dialog

        # Initialize the first speaker's label and line
        current_label, current_lines = dialog[0]

        for label, line in dialog[1:]:
            if label == current_label:
                # If the speaker is the same, append the line to the current lines
                current_lines += " " + line
            else:
                # If the speaker changes, add the current speaker and their lines to the result
                merged_dialog.append((current_label, current_lines))
                # Update to the new speaker
                current_label, current_lines = label, line

        # Don't forget to add the last speaker's lines
        merged_dialog.append((current_label, current_lines))

        # Tag empty lines as UNINTELLIGIBLE
        marked_dialog = []
        for label, speech in merged_dialog:
            if speech == "":
                marked_dialog.append((label, "<UNINTELLIGIBLE>"))
            else:
                marked_dialog.append((label, speech))

        return marked_dialog

    @staticmethod
    def format_turn_taking(dialog: list[tuple[str, str]]) -> list[tuple[str, str, str]]:  # noqa: C901
        """Formating the dialog into turn-taking format."""
        formatted_dialog = []
        child_line = None
        adult_line = None
        adult_label = None

        for idx, (label, line) in enumerate(dialog):
            # We encounter child speech and have a previous adult speech registered
            if label == "CHI" and adult_label is not None:
                formatted_dialog.append((adult_label, adult_line, line))
                # reset registers
                adult_label, adult_line, child_line = None, None, None

            # We encounter child speech but no previous adult speech exists
            elif label == "CHI" and adult_label is None:
                if child_line is not None:
                    raise ValueError(f"illegal child consecutive speech item: {idx}")
                child_line = line

            # Non keyCHILD speech, with previously registered child speech
            elif label != "CHI" and child_line is not None:
                formatted_dialog.append((label, line, child_line))
                # reset registers
                adult_label, adult_line, child_line = None, None, None

            # Non keyCHILD speech, without previously registered child speech
            elif label != "CHI" and child_line is None:
                # Previous speaker is also Non keyCHIL
                if adult_label:
                    formatted_dialog.append((adult_label, adult_line, "<EMPTY>"))
                # update registers
                adult_label, adult_line = label, line

        # pushing odd leftovers on registries
        if child_line and adult_line and adult_label:
            formatted_dialog.append((adult_label, adult_line, child_line))
        elif child_line and adult_line is None:
            formatted_dialog.append(("-", "<EMPTY>", child_line))
        elif child_line is None and adult_line and adult_label:
            formatted_dialog.append((adult_label, adult_line, "<EMPTY>"))

        return formatted_dialog

    @classmethod
    def turn_taking_mk(cls, file: Path) -> list[TurnTakeData]:
        """Convert given file into the turn-taking format."""
        dialog = json.loads(file.read_bytes())
        merged_dialog = cls.merge_consecutive_speakers(dialog)
        turn_taking_dialog = cls.format_turn_taking(merged_dialog)
        # Return as turn-taking data
        return [TurnTakeData(*rows) for rows in turn_taking_dialog]
