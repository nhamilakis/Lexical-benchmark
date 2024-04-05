import collections
from pathlib import Path
import typing as t

from ..utils import extract_from_cha


class CHAList:
    """ An object interfacing a subset of  CHA files """

    @property
    def transcription_data(self):
        if self.data is None:
            self.scrap_transcriptions()
        return self.data

    @property
    def words_list(self) -> tuple[list[str], list[str]]:
        if None in (self._adult_word_list, self._child_word_list):
            self.build_words_lists()
        return self._adult_word_list, self._child_word_list

    def __init__(self, annotations: list[Path]):
        self.annotations = annotations
        self.data = None
        self._adult_word_list = None
        self._child_word_list = None

    def scrap_transcriptions(self):
        """ Extract & format transcriptions from CHA files """
        self.data = dict()

        for annotation in self.annotations:
            self.data[annotation.name] = extract_from_cha(annotation)

    def build_words_lists(self):
        """ Extract words from CHA transcriptions """
        child_words = list()
        adult_words = list()

        for _, cha in self.transcription_data.items():
            child_words.extend(cha.child_tr.raw_words())
            adult_words.extend(cha.adult_tr.raw_words())

        self._adult_word_list = adult_words
        self._child_word_list = child_words


class CHILDES:
    """
        Use Jing's version of CHILDES (on oberon).

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

    @property
    def parsed_transcriptions(self):
        if self._parsed_transcriptions is None:
            self.parse_transcriptions()
        return self._parsed_transcriptions

    def __init__(self, root_dir: Path):
        if not root_dir.is_dir():
            raise ValueError(f'Given directory :: {root_dir} does not exist !!')

        self.root_dir = root_dir
        self._parsed_transcriptions = None

    def language_code_items(self) -> t.Iterable[str]:
        """ Iterable containing the list of language codes  """
        yield from (lang.name for lang in (self.root_dir / 'transcript').iterdir())

    def corpus_items(self) -> t.Iterable[tuple[str, str]]:
        """ Iterable containing a tuple typed: (land_code, corpus_path)"""
        for lang in self.language_code_items():
            yield from ((lang, corpus.name) for corpus in (self.root_dir / lang).iterdir())

    def annotations_items(self) -> t.Iterable[str, str, list[Path]]:
        for lang, corpus in self.corpus_items():
            yield from (
                (lang, corpus, list((self.root_dir / lang / corpus).rglob("*.cha")))
            )

    def parse_transcriptions(self):
        """ Extract all transcriptions from the given CHILDES dataset """
        data = collections.defaultdict(dict)
        for lang, corpus, annotations in self.annotations_items():
            data[lang][corpus] = CHAList(annotations)
        self._parsed_transcriptions = dict(data)

    def word_frequencies(self, lang: str) -> tuple[collections.Counter, collections.Counter]:
        """ Calculate word frequency total"""
        adult_total = list()
        child_total = list()
        corpus_dict = self.parsed_transcriptions.get(lang, {})

        for _, cha_lst in corpus_dict.items():
            adult, child = cha_lst.words_list()
            adult_total.extend(adult)
            child_total.extend(child)

        return collections.Counter(adult_total), collections.Counter(child_total)
