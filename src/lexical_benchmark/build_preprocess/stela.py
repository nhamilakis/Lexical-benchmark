"""Tools to collect and use transcriptions from the STELA model."""

import dataclasses
import typing as t
from pathlib import Path

import pandas as pd
import polars as pl

from lexical_benchmark import datasets, settings
from lexical_benchmark.dataloaders import preprocess as preprocess_dataloader


@dataclasses.dataclass
class AudioRow:
    """The typing of a row mapping STELA Audio files."""

    language: str
    hour: str  # hour split
    split: str  # speaker split
    speaker: str  # the speaker id
    book: str  # the book id
    wav: str  # Name of the file

    @property
    def values(self) -> tuple[str, ...]:
        """Get Values of row."""
        return tuple(dataclasses.asdict(self).values())

    def rel_path(self, dataset_root: Path | None = None) -> Path:
        """Get path relative to a dataset root."""
        if dataset_root is None:
            dataset_root = Path()

        return dataset_root / self.language / self.hour / self.split / self.speaker / self.book / self.wav


class BookTxt(t.NamedTuple):
    """Typing for the Book / Transcription file association."""

    book: str
    text: str

    def val(self) -> tuple[str, str]:
        """Extract values."""
        return self.book, self.text


class InfTrainStructure:
    """Manipulate InfTrain File Structure."""

    @property
    def transcription_location(self) -> Path:
        """Return the location of the transcriptions."""
        return self.dataset_dir / "text" / f"{self.lang}" / "LibriVox"

    @property
    def transciptions_iter(self) -> t.Iterable[Path]:
        """An Iterable on all the english transcriptions."""
        yield from (self.dataset_dir / f"text/{self.lang}/LibriVox").rglob("*.txt")

    @property
    def en_audio_iter(self) -> t.Iterable[Path]:
        """An Iterable on all the english audio files."""
        yield from (self.dataset_dir / f"wav/{self.lang}").rglob("*.wav")

    @property
    def tree_root(self) -> Path:
        """Get root dir of tree architecture."""
        return self.dataset_dir / "symlinks" / self.lang.upper()

    @property
    def hours_split(self) -> tuple[str, ...]:
        """Get a tuple of the data-split in hours."""
        return tuple(d.name for d in self.tree_root.iterdir() if d.is_dir())

    def split_codes(self, hour: str) -> tuple[str, ...]:
        """Get a tuple of all the speaker_codes in the specified hour-split."""
        return tuple(d.name for d in (self.tree_root / hour).iterdir() if d.is_dir())

    def speaker_ids(self, hour: str, split: str) -> tuple[str, ...]:
        """Get a tuple of all speaker_ids in a specified hour/speaker set."""
        return tuple(d.name for d in (self.tree_root / hour / split).iterdir() if d.is_dir())

    def book_ids(self, hour: str, split: str, speaker: str) -> tuple[str, ...]:
        """Get the tags contained in a hour/sp/speaker set."""
        return tuple(d.name for d in (self.tree_root / hour / split / speaker).iterdir() if d.is_dir())

    def wavs(self, hour: str, split: str, speaker: str, book: str) -> tuple[str, ...]:
        """Get the wav files contained in a hour/speaker/book/tag set."""
        return tuple(d.name for d in (self.tree_root / hour / split / speaker / book).glob("*.wav"))

    def iter_wavs(self) -> t.Iterable[AudioRow]:
        """An iterable over all the architecture of wavs."""
        for hour in self.hours_split:
            for split in self.split_codes(hour):
                for speaker in self.speaker_ids(hour, split):
                    for book in self.book_ids(hour, split, speaker):
                        yield from [
                            AudioRow(language=self.lang, hour=hour, split=split, speaker=speaker, book=book, wav=wav)
                            for wav in self.wavs(hour, split, speaker, book)
                        ]

    def wav_split_associations(self) -> pl.DataFrame:
        """Build wav associations DataFrame."""
        return pl.DataFrame([dataclasses.asdict(row) for row in self.iter_wavs()])

    def matched_metadata(self) -> pl.DataFrame:
        """Clean matched2.csv to keep only usefull items."""
        # Load matched file & keep only current language
        matched = pl.read_csv(self.matched_metadata_file, infer_schema_length=False)

        def only_fname(p: str) -> str:
            """Remove uselless path part."""
            return Path(p).name

        # Remove unwanted columns
        return (
            # lowercase the language
            matched.filter(pl.col("language") == self.lang.lower())
            # remove absolute path of audio & text path
            .with_columns(
                [
                    pl.col("text_path").map_elements(only_fname).alias("text_path"),
                    pl.col("audio_path").map_elements(only_fname).alias("audio_path"),
                ]
            )
            .rename({"book_id": "book"})
            # keep only relevant columns
            .select(["text_path", "book", "genre", "book_title", "text_source"])
        )

    def wav_text_associations(self) -> pl.DataFrame:
        """Build text/wav associations DataFrame."""
        assoc = self.wav_split_associations()
        matched = self.matched_metadata()
        # Keep one ref per book (drop duplicates)
        matched = matched.unique(subset=["book"])
        # Merge by book ID
        return assoc.join(matched, on="book")

    def __init__(self, root_dir: Path, metadata_file: Path, lang: str = "en") -> None:
        self.dataset_dir = root_dir
        self.lang = lang.upper()
        self.matched_metadata_file = metadata_file


class STELAPrepTranscripts:
    """Class used to manipulate STELA transcripts."""

    @property
    def associations_file(self) -> Path:
        """Path to target associations file."""
        return self.dataset_cfg.meta_dir / self.lang / "associations.csv"

    def __init__(
        self,
        lang: str,
        bad_books: tuple[str, ...] = (),
        *,
        use_asr: bool = False,
    ) -> None:
        self.dataset_cfg: datasets.STELADatasetConfig = datasets.get_config("stela")
        self.lang = lang.upper()
        self.inf_train = InfTrainStructure(
            root_dir=self.dataset_cfg.original_root,
            metadata_file=self.dataset_cfg.source_matched_csv,
            lang=lang,
        )
        self.BAD_BOOKS = bad_books
        # preset empty items
        self.books: dict[str, Path] = {}
        self.asr_location = self.dataset_cfg.asr_books_path if use_asr else None

    def make_source(self) -> None:
        """Creates corresponding symlinks to create the dataset."""
        self.dataset_cfg.original_root.parent.mkdir(exist_ok=True, parents=True)
        try:
            if not self.dataset_cfg.original_root.is_symlink():
                self.dataset_cfg.original_root.symlink_to(settings.PATH.stela_original)
        except SystemError as err:
            raise ValueError("Could not locate SOURCE of STELATranscriptDataset") from err

    def associations_df(self) -> pd.DataFrame:
        """Load asscociations as a DataFrame."""
        if self.associations_file.is_file():
            return pl.read_csv(self.associations_file, sep=";")

        if not self.associations_file.parent.is_dir():
            self.associations_file.parent.mkdir(parents=True)

        # If asscociations were not build make them from infTrain dataset
        associations = self.inf_train.wav_text_associations()
        # Save the file
        associations.write_csv(str(self.associations_file), include_header=True, separator=";")
        return associations

    def build_book_dict(self) -> None:
        """Build dictionairy with bookname to file association."""
        if len(self.books) > 0:
            return

        associations = self.associations_df()[["book", "text_path"]]
        associations = associations.drop_duplicates(subset=["book"], keep="first")
        associations["text_path"] = associations["text_path"].apply(
            lambda tp: self.inf_train.transcription_location / tp
        )

        # note: itertuples recognised as tuple[any, ... ] instead of tuple[str, str] required by dict
        book_id_dict: dict[str, Path] = dict(associations.itertuples(index=False, name=None))  # type: ignore[arg-type,annotation-unchecked]
        self.books = book_id_dict

    def get_asr(self, book_id: str) -> str:
        """Load book text from ASR transcriptions."""
        if self.asr_location is None:
            raise ValueError("ASR location not specified.")

        print(f"Using ASR: for  {book_id}.")
        return "\n".join([file.read_text() for file in (self.asr_location / book_id).glob("*.txt")])

    def extract_book_content(self, book_list: list[str]) -> dict[str, str]:
        """Write a book list into a single file."""
        # build book index
        self.build_book_dict()
        result = {}
        for book in book_list:
            text_path = self.books.get(book, None)
            if text_path is None:
                raise ValueError(f"Not found {book}")

            if self.asr_location and book in self.BAD_BOOKS:
                result += self.get_asr(book) + " "
            else:
                result[book] = text_path.read_text()
        return result

    def iter_transcriptions_by_split(self) -> t.Iterable[tuple[str, str, list[str]]]:
        """Load transcriptions by split category."""
        associations = self.associations_df()[["hour", "split", "book"]]
        # this is bad ::: associations = associations.drop_duplicates(subset=["book"], keep="first")

        # note: as usual something in pandas does not type correctly
        associations = associations.groupby(["hour", "split"], as_index=False)["book"].agg(",".join)  # type: ignore[assignment]

        for row in associations.itertuples(index=False):
            # Keep only one copy of each book
            booklist = list(set(str(row.book).split(",")))
            yield f"{row.hour}", f"{row.split:02}", booklist

    def tag_asr_books(self, book_list: list[str]) -> list[str]:
        """Add ASR tag to bad books."""

        def tag(book: str) -> str:
            if book in self.BAD_BOOKS:
                return f"{book}/asr"
            return book

        return [tag(book) for book in book_list]

    def build_preprocess(self) -> None:
        """Make train folder architecture."""
        for hour, split, booklist in self.iter_transcriptions_by_split():
            item = preprocess_dataloader.STELAPreprocessedItems(lang=self.lang, hour_split=hour, chunk=split)
            # Fetch book transcriptions
            text_dict = self.extract_book_content(booklist)
            # Write transcriptions
            for book, transcription in text_dict.items():
                (item.root_dir / "books" / f"{book}.raw").safe_write_text(transcription)
