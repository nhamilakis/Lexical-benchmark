"""Tools to collect and use transcriptions from the STELA model."""

import dataclasses
import typing as t
import warnings
from pathlib import Path

from lexical_benchmark import settings

from .data import STELATranscriptDataset

warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd  # noqa: E402 (Deprecation avoid)


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
    def matched_metadata_file(self) -> Path:
        """Return the file allowing to match audio to transcription."""
        return self.dataset_dir / "metadata/matched2.csv"

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

    def wav_split_associations(self) -> pd.DataFrame:
        """Build wav associations DataFrame."""
        return pd.DataFrame.from_records([dataclasses.asdict(row) for row in self.iter_wavs()])

    def matched_metadata(self) -> pd.DataFrame:
        """Clean matched2.csv to keep only usefull items."""
        # Load matched file & keep only current language
        matched = pd.read_csv(str(self.matched_metadata_file), header=0)
        matched = matched[matched["language"] == self.lang.lower()]

        def only_fname(p: str) -> str:
            """Remove uselless path part."""
            return Path(p).name

        # Remove unnecessary path folders
        matched.loc[:, "text_path"] = matched["text_path"].apply(only_fname)
        matched.loc[:, "audio_path"] = matched["audio_path"].apply(only_fname)

        # Rename to match wav associations
        matched = matched.rename(columns={"book_id": "book"})

        # Remove unwanted columns
        return matched[["text_path", "book"]]

    def wav_text_associations(self) -> pd.DataFrame:
        """Build text/wav associations DataFrame."""
        assoc = self.wav_split_associations()
        matched = self.matched_metadata()
        # Keep one ref per book
        matched = matched.drop_duplicates(subset=["book"])
        # Merge by book ID
        return assoc.merge(matched, on="book")

    def __init__(self, root_dir: Path, metadata_dir: Path, lang: str = "en") -> None:
        self.dataset_dir = root_dir
        self.metadata_dir = metadata_dir
        self.lang = lang.upper()


class STELAPrepTranscripts:
    """Class used to manipulate STELA transcripts."""

    @property
    def associations_file(self) -> Path:
        """The file storing Wav/Text Associations."""
        return self.dataset.meta.wav_text_associations

    @property
    def train_dir(self) -> Path:
        """Location to store train dataset."""
        return self.target_dir / "train"

    def __init__(
        self,
        lang: str,
        root_dir: Path = settings.PATH.stela,
        with_asr: Path | None = None,
        bad_books: tuple[str, ...] = (),
    ) -> None:
        self.dataset = STELATranscriptDataset(root_dir=root_dir)
        self.target_dir = self.dataset.preprocessed_path
        self.lang = lang.upper()
        self.inf_train = InfTrainStructure(
            root_dir=self.dataset.source_path,
            metadata_dir=self.dataset.meta.meta_root_path,
            lang=lang,
        )
        self.BAD_BOOKS = bad_books
        # preset empty items
        self.books: dict[str, Path] = {}
        self.asr_location = with_asr

    def make_source(self) -> None:
        """Creates corresponding symlinks to create the dataset."""
        self.dataset.source_path.parent.mkdir(exist_ok=True, parents=True)
        try:
            if not self.dataset.source_path.is_symlink():
                self.dataset.source_path.symlink_to(settings.PATH.stela_original)
        except SystemError as err:
            raise ValueError("Could not locate SOURCE of STELATranscriptDataset") from err

    def associations_df(self) -> pd.DataFrame:
        """Load asscociations as a DataFrame."""
        if self.associations_file.is_file():
            return pd.read_csv(str(self.associations_file), sep=";")

        if not self.associations_file.parent.is_dir():
            self.associations_file.parent.mkdir(parents=True)

        # If asscociations were not build make them from infTrain dataset
        associations = self.inf_train.wav_text_associations()
        # Save the file
        associations.to_csv(str(self.associations_file), index=False, sep=";")
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

    def merged_book_transcriptions(self, book_list: list[str]) -> str:
        """Write a book list into a single file."""
        # build book index
        self.build_book_dict()
        result = ""
        for book in book_list:
            text_path = self.books.get(book, None)
            if text_path is None:
                raise ValueError(f"Not found {book}")

            if self.asr_location and book in self.BAD_BOOKS:
                result += self.get_asr(book) + " "
            else:
                result += text_path.read_text() + " "
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

    def build_transcript(self, root_dir: Path | None = None) -> None:
        """Make train folder architecture."""
        c_root_dir = root_dir if root_dir is not None else self.target_dir / "txt"

        c_root_dir = c_root_dir / self.lang
        for hour, split, booklist in self.iter_transcriptions_by_split():
            item = self.dataset.item(lang=self.lang, hour=hour, section=split)
            # Fetch book transcriptions
            text = self.merged_book_transcriptions(booklist)
            # Write transcriptions
            item.preprocess.raw.safe_write_text(text)
            # Write list of books used for transcription
            item.clean.books.safe_write_text("\n".join(self.tag_asr_books(booklist)))
