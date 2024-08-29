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
    def add_word(cls, label: str, word: str) -> None:
        """Save a clean word to the Log."""
        cls._LOGS[label].append(word)

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
    def dump_logs(cls, file: Path) -> None:
        """Dump all logs into a file."""
        logs = cls.export_logs()

        with file.open("w") as fh:
            json.dump(logs, fh, indent=4)

        # Remove all items
        cls._COUNT_LOGS.clear()
        cls._LOGS.clear()


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


class NumberFixer(TextActionFN):
    """Remove hanging numbers in text."""

    def __init__(self, *, keep_as_text: bool = True) -> None:
        super().__init__(label="numbers")
        self.keep_as_text = keep_as_text
        self.pattern = re.compile(r"\b\d+\b")

    def __call__(self, line: str) -> str:
        """Clean line of numbers."""
        matches = self.pattern.findall(line)

        clean_line = line
        if self.keep_as_text:
            for m in matches:
                as_lang = num2words(m)
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
