import abc
import collections
import contextlib
import json
import re
import string
import typing as t
import unicodedata
from pathlib import Path

from num2words import num2words

from .various import to_roman


class CleanerFN(t.Protocol):
    """Protocol for clean function-class."""

    def __call__(self, line: str) -> str:
        """Run clean."""


def piped(line: str, *fn_list: CleanerFN) -> str:
    """Piping function for chaining text cleaning rulesets."""
    cl_line = line
    for fn in fn_list:
        cl_line = fn(cl_line)
    return cl_line


class WordLogger:
    """Class that allows logging of words."""

    _LOGS: t.ClassVar[dict[str, list[str]]] = collections.defaultdict(list)
    _COUNT_LOGS: t.ClassVar[dict[str, int]] = collections.defaultdict(lambda: 0)
    _ERROR_LOG: t.ClassVar[dict[str, list[str]]] = collections.defaultdict(list)

    @classmethod
    def add_word(cls, label: str, *word: str) -> None:
        """Save a clean word to the Log."""
        cls._LOGS[label].extend(word)

    @classmethod
    def add_error(cls, label: str, msg: str) -> None:
        """Save a clean word to the Log."""
        cls._ERROR_LOG[label].append(msg)

    @classmethod
    def update_log(cls, label: str, nb: int = 1) -> None:
        """Add one to the count of an action."""
        cls._COUNT_LOGS[label] += nb

    @classmethod
    def get_log(cls, label: str) -> list[str] | int:
        """Extract a log by category."""
        if label in cls._LOGS:
            return cls._LOGS[label]
        return cls._COUNT_LOGS[label]

    @classmethod
    def export_logs(cls) -> dict[str, list[str] | int]:
        """Export logs as dictionairy."""
        return {**dict(cls._LOGS), **dict(cls._COUNT_LOGS)}

    @classmethod
    def dumps_logs(cls) -> dict:
        """Dump all logs as dict."""
        logs = cls.export_logs()

        # Reset all items
        cls._COUNT_LOGS.clear()
        cls._LOGS.clear()

        return logs

    @classmethod
    def dump_logs(cls, file: Path) -> None:
        """Dump all logs into a json file."""
        logs = cls.dumps_logs()

        with file.open("w") as fh:
            json.dump(logs, fh, indent=4)


class TextActionFN(WordLogger, abc.ABC):
    """Abstract text clean action."""

    def __init__(self, label: str) -> None:
        self.label = label

    @abc.abstractmethod
    def __call__(self, line: str) -> str:
        """Use the action on a given line."""

    def __str__(self) -> str:
        """Show Action class."""
        return f"[{self.__class__} at {id(self):#X}, rule:: {self.label}]"

    def __repr__(self) -> str:
        """Show Action class."""
        return self.__str__()


class TextNormalization(TextActionFN):
    """Normalise Text, for processing."""

    def __init__(self) -> None:
        super().__init__(label="Normalise TXT")

    def rmdiacritics(self, char: str) -> str:
        """Normalise char, by "removing" any diacritics like accents or curls and strokes and the like."""
        try:
            desc = unicodedata.name(char)
        except ValueError:
            return ""

        cutoff = desc.find(" WITH ")
        if cutoff != -1:
            desc = desc[:cutoff]
            with contextlib.suppress(KeyError):
                char = unicodedata.lookup(desc)
        return char

    def __call__(self, line: str) -> str:
        """Normalise a line of text by fixing diacritics & removing all bad characters."""
        line_normalised = "".join(map(self.rmdiacritics, line))
        return "".join(filter(lambda x: x in string.printable, line_normalised))


class QuotationCleaner(TextActionFN):
    """Clean single Quotted words."""

    def __init__(self) -> None:
        super().__init__(label="Normalise Quotations")
        self.pattern = re.compile(r"'\b([^']+)\b'")

    def __call__(self, line: str) -> str:
        """Clean quoted text that uses single lines."""
        # Remove weird consecutive double quotes
        line = line.replace("''", "")

        matches = self.pattern.findall(line)
        for m in matches:
            line = line.replace(f"'{m}'", f"{m}")

        return line


class SpecialCharacterTranscriptions(TextActionFN):
    """Replaces special characters with their transcribed mode."""

    def __init__(self, lang: str, *, keep: bool = True) -> None:
        super().__init__(label="SpecialCharacterTranscription")
        self.keep = keep
        self.lang = lang

    def transcribe_form(self, c: str) -> str:
        """Get transcribe form of character."""
        res: str | None = {
            "EN": {
                "$": "dollar",
                "€": "euro",
                "&": "and",
            }
        }.get(self.lang, {}).get(c)

        if self.keep and res is not None:
            return res
        return " "

    def __call__(self, line: str) -> str:
        """Replace all special characters with their transcribed mode."""
        line = line.replace("$", f" {self.transcribe_form('$')} ")
        line = line.replace("@", f" {self.transcribe_form('@')} ")
        line = line.replace("€", f" {self.transcribe_form('€')} ")
        line = line.replace("&", f" {self.transcribe_form('&')} ")

        return line  # noqa: RET504


class IllustrationRemoval(TextActionFN):
    """Removes the [illustation] tag from text."""

    def __init__(self) -> None:
        super().__init__(label="illustration-removal")
        self.tag = re.compile(r"\[Illustration([^\]]*)\]")

    def __call__(self, line: str) -> str:
        """Removes the [illustration] tag."""
        count = len(self.tag.findall(line))
        self.update_log(self.label, count)

        # Replace illustration tags in line
        return self.tag.sub("", line)


class NumberFixer(TextActionFN):
    """Remove hanging numbers in text."""

    def __init__(self, *, keep_as_text: bool = True) -> None:
        super().__init__(label="numbers")
        self.keep_as_text = keep_as_text
        self.pattern = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")

    def __call__(self, line: str) -> str:
        """Clean line of numbers."""
        matches = self.pattern.findall(line)

        clean_line = line
        if self.keep_as_text:
            for m in matches:
                as_lang = num2words(m.replace(",", ""))
                self.add_word(self.label, f"{m}::{as_lang}")
                clean_line = clean_line.replace(m, as_lang)
        else:
            for m in matches:
                self.add_word(self.label, m)
                clean_line = clean_line.replace(m, "")

        return clean_line


class PatternRemover(TextActionFN):
    """A class to help remove patterns."""

    def __init__(self, *, match: t.Pattern[str], label: str, subst: str = "") -> None:
        super().__init__(label=label)
        self.pattern = match
        self.subst = subst

    def __call__(self, line: str) -> str:
        """Run clean Operation."""
        matches = self.pattern.findall(line)

        # Register Words
        for m in matches:
            self.add_word(self.label, m)

        # Return line without matches
        return self.pattern.sub(self.subst, line)


class RomanNumerals(TextActionFN):
    """A class to remove roman numerals.

    Note:
    ----
        This ignores number 1 because its impossible to differenciate from the capital i.

    """

    def __init__(self, max_number: int = 1000) -> None:
        super().__init__(label="ROMAN_NUMERALS")
        self.map = {to_roman(x).lower() for x in range(2, max_number)}

    def __call__(self, line: str) -> str:
        """Clean given from roman numerals."""
        clean_line = [word for word in line.split() if word.lower() not in self.map]
        return " ".join(clean_line)


class URLRemover(TextActionFN):
    """A class to remove URLs from text."""

    def __init__(self) -> None:
        super().__init__(label="URL")
        self.url_match = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

    def __call__(self, line: str) -> str:
        """Clean line from URLs."""
        # Log URLs before removing
        urls_found = self.url_match.findall(line)
        self.add_word(self.label, *urls_found)

        # Return cleaned line
        return self.url_match.sub("", line)


class AZFilter(TextActionFN):
    """Filters text using an AZ filter.

    Notes
    -----
        This filter removes any characters that are not in the following list
        - letters between a-z
        - the apostrophe character (to preserve context in words like ain't)
        - the '-' character to preserve complex words like fifty-six
        - the space character to preserve separation of words
        After the filter has been applied the result text is lowecased.

    """

    def __init__(self) -> None:
        super().__init__(label="AlphabeticFilter")
        # Append apostrophe & space to the allowed chars as to not break words
        self.allowed_chars = string.ascii_lowercase + "-' "

    def __call__(self, line: str) -> str:
        """Clean current line to keep only pure text."""
        unclean_chars = "".join({c.lower() for c in line if c.lower() not in self.allowed_chars})
        self.add_word(self.label, unclean_chars)

        clean_line = "".join(c for c in line if c.lower() in self.allowed_chars).lower()
        # Replace hyphen with space
        return clean_line.replace("-", " ")


class WordCleaner(TextActionFN):
    """Perform inter word cleaning.

    Rules:
    -----
    1) Quoted words
    Matches any word containing a ' (quote) symbol.
    This mostly referers to shorthands for ex:
    - ain't
    - can't
    - don't
    - etc..
    These words are cleaned, indexed and merged.

    2) Hyphenated words
    Matches complex words containing a hyphen
    - fifty-six
    - ...

    These words are indexed, cleaned by removing the hyphen ?

    """

    def __init__(self) -> None:
        super().__init__(label="word_cleaning")

    def __call__(self, line: str) -> str:
        """Clean all the words in the given line."""
        clean_words = []
        for word in line.split():
            w = word
            if "'" in w:
                w = w.replace("'", "")
                self.add_word("quotted_words", w)

            if "-" in w:
                self.add_word("hyphen-words", w)
                w = w.replace("-", " ")

            clean_words.append(w)

        return " ".join(clean_words)


class CharSeqRemover(TextActionFN):
    """A class to help removing chars sequences from text."""

    def __init__(self, seq: str, label: str, *, subst: str = "", count: bool = False) -> None:
        super().__init__(label=label)
        self.seq = seq
        self.subst = subst
        self.count = count

    def __call__(self, line: str) -> str:
        """Clean given line from specific chars."""
        if self.count:
            self.update_log(self.label, nb=line.count(self.seq))
        return line.replace(self.seq, self.subst)


class MultiCharSeqRemover(CharSeqRemover):
    """A class to help removing multiple chars sequences from text."""

    def __init__(self, *char_seqs: str, label: str, subst: str = "", count: bool = False) -> None:
        super().__init__(" ", label, subst=subst, count=count)
        self.char_seq_list = char_seqs

    def __call__(self, line: str) -> str:
        """Clean line from all sequences."""
        clean_line = line
        for seq in self.char_seq_list:
            self.seq = seq
            clean_line = super().__call__(clean_line)
        return clean_line
