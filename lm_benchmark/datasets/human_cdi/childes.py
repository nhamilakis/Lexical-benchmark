import collections
import typing as t
from pathlib import Path

from lm_benchmark.datasets import utils


class CHAList:
    """An object interfacing a subset of  CHA files."""

    def __init__(self, annotations: list[Path]) -> None:
        self.annotations = annotations
        self.data: dict[str, utils.CHAData] = {}
        self._adult_word_list: list[str] | None = None
        self._child_word_list: list[str] | None = None

    @property
    def transcription_data(self) -> dict[str, utils.CHAData]:
        """Return transcription data."""
        if len(self.data) <= 0:
            self.data.update(self.scrap_transcriptions())
        return self.data

    @property
    def words_list(self) -> tuple[list[str], list[str]]:
        """Return word list."""
        if self._adult_word_list is None or self._child_word_list is None:
            adult_words, child_words = self.build_words_lists()
            self._adult_word_list = adult_words
            self._child_word_list = child_words
        return self._adult_word_list, self._child_word_list

    def scrap_transcriptions(self) -> dict[str, utils.CHAData]:
        """Extract & format transcriptions from CHA files."""
        data = {}

        for annotation in self.annotations:
            data[annotation.name] = utils.extract_from_cha(annotation)
        return data

    def build_words_lists(self) -> tuple[list[str], list[str]]:
        """Extract words from CHA transcriptions."""
        child_words, adult_words = [], []

        for cha in self.transcription_data.values():
            child_words.extend(cha.child_tr.raw_words())
            adult_words.extend(cha.adult_tr.raw_words())

        return adult_words, child_words


class CHILDES:
    """Use Jing's version of CHILDES (on oberon).

    Data Structure:

    - lang-code {NA UK, ...}
        - text {NA English, ...}

    extract from annotation CHI,ADULT(all) speech
    clean up all things that are not words.

    Create  a CSV file containing a list of utterances

    Output:

    - child, speaker, language, corpus, number_of_tokens, src_path, filename, content
    - Word Frequency table
    """

    def __init__(self, root_dir: Path) -> None:
        if not root_dir.is_dir():
            raise ValueError(f"Given directory :: {root_dir} does not exist !!")
        self.root_dir = root_dir
        self._parsed_transcriptions: dict[str, dict[str, CHAList]] | None = None

    @property
    def parsed_transcriptions(self) -> dict[str, dict[str, CHAList]]:
        """Fetch parsed transcriptions (build them if not found)."""
        if self._parsed_transcriptions is None:
            self._parsed_transcriptions = self.mk_parsed_transcriptions()
        return self._parsed_transcriptions

    def language_code_items(self) -> t.Iterable[str]:
        """Iterable containing the list of language codes."""
        yield from (lang.name for lang in (self.root_dir / "transcript").iterdir())

    def corpus_items(self) -> t.Iterable[tuple[str, str]]:
        """Iterable containing a tuple typed: (land_code, corpus_path)."""
        for lang in self.language_code_items():
            yield from ((lang, corpus.name) for corpus in (self.root_dir / lang).iterdir())

    def annotations_items(self) -> t.Iterable[str, str, list[Path]]:
        """Iterate over annotation items."""
        for lang, corpus in self.corpus_items():
            yield from ((lang, corpus, list((self.root_dir / lang / corpus).rglob("*.cha"))))

    def mk_parsed_transcriptions(self) -> dict[str, dict[str, CHAList]]:
        """Extract all transcriptions from the given CHILDES dataset."""
        data: collections.defaultdict[str, dict[str, CHAList]] = collections.defaultdict(dict)
        for lang, corpus, annotations in self.annotations_items():
            data[lang][corpus] = CHAList(annotations)
        return dict(data)

    def word_frequencies(self, lang: str) -> tuple[collections.Counter, collections.Counter]:
        """Calculate word frequency total."""
        adult_total, child_total = [], []
        corpus_dict: dict[str, CHAList] = self.parsed_transcriptions.get(lang, {})

        for cha_lst in corpus_dict.values():
            adult, child = cha_lst.words_list
            adult_total.extend(adult)
            child_total.extend(child)

        return collections.Counter(adult_total), collections.Counter(child_total)
