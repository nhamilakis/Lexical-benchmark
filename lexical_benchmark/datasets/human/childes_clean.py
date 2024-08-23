import collections
import functools
import json
import re
import typing as t
from datetime import date
from pathlib import Path

CLEANER_TYPE = t.Callable[[str], tuple[str, int]]


class WordLogger:
    """Class that allows logging of words."""

    _LOGS: t.ClassVar[dict[str, list[str]]] = collections.defaultdict(list)

    @classmethod
    def add_word(cls, label: str, word: str) -> None:
        """Save a clean word to the Log."""
        cls._LOGS[label].append(word)

    @classmethod
    def get_log(cls, label: str) -> list[str]:
        """Extract a log by category."""
        return cls._LOGS[label]

    @classmethod
    def export_logs(cls) -> dict[str, list[str]]:
        """Export logs as dictionairy."""
        return dict(cls._LOGS)


class TagCleaner(WordLogger):
    """A class helper for cleaning CHILDES tags."""

    def __init__(self, *, clean_pattern: str, match_pattern: t.Pattern[str], label: str, clean: bool = True) -> None:
        self.clean_pattern = clean_pattern
        self.match_pattern = match_pattern
        self.label = label
        self.clean = clean

    def __call__(self, line: str) -> str:
        """Run clean Operation."""
        matches = self.match_pattern.findall(line)

        for m in matches:
            self.add_word(m.replace(self.clean_pattern, ""), self.label)

        # Return line cleanned of tagged words
        return line.replace(self.clean_pattern, "")


def pattern_remover(*, patern: t.Pattern[str], replace_with: str = "") -> CLEANER_TYPE:
    """Removes given pattern from given string."""

    def _internal_fn(line: str, *, _pattern: t.Pattern[str], _replace_with: str = replace_with) -> tuple[str, int]:
        return _pattern.sub(_replace_with, line), len(_pattern.findall(line))

    # Return as a partial rule
    return functools.partial(_internal_fn, _pattern=patern, _replace_with=replace_with)


def substring_remover(*, seq: str, replace_with: str = "") -> CLEANER_TYPE:
    """Replace sequence of characters from given string."""

    def _internal_fn(line: str, _seq: str, _replace_with: str = "") -> tuple[str, int]:
        return line.replace(_seq, _replace_with), line.count(_seq)

    return functools.partial(_internal_fn, _seq=seq, _replace_with=replace_with)


def word_remover(*, pattern: t.Pattern[str]) -> CLEANER_TYPE:
    """Remove a word from the coprus."""
    return pattern_remover(patern=pattern, replace_with=" ")


cleaning_adult_speech_rules: list[tuple[CLEANER_TYPE, str]] = [
    # Remove Bracket  ([text]) Annotations
    (pattern_remover(patern=re.compile(r"\[([^\]]+)\]")), "bracket-annotation"),
    # Remove Paranthesis Annotations
    (pattern_remover(patern=re.compile(r"\s\(([^()]*?)\)\s")), "parenthesis-annotation"),
    # Onomatopoeia (@o): KEEP
    (substring_remover(seq="@o"), "@o"),
    # Phonological consistent forms (@p): KEEP
    (substring_remover(seq="@p"), "@p"),
    # Babbling (@b): Discard
    (word_remover(pattern=re.compile(r"\s[^ ]*@b\s?")), "@b"),
    # Word-Play (@wp): Discard
    (word_remover(pattern=re.compile(r"\s[^ ]*@wp\s?")), "@wp"),
    # Child Invented Form (@c): Discard
    (word_remover(pattern=re.compile(r"\s[^ ]*@c\s?")), "@c"),
    # Family Specific Form (@f): Discard
    (word_remover(pattern=re.compile(r"\s[^ ]*@c\s?")), "@c"),
    # Dialect Word (@d): KEEP
    (substring_remover(seq="@d"), "@d"),
    # TODO(@nhamilakis): handle Second (or other) Language (@s:...)
    # Neologism (@n): KEEP
    (substring_remover(seq="@n"), "@n"),
    # Singing (@si): KEEP
    (substring_remover(seq="@si"), "@si"),
    # Interjection/interaction (@i): KEEP
    (substring_remover(seq="@si"), "@si"),
    # Remove Test Words (@t) annotation
    (substring_remover(seq="@t"), "@t"),
    # Meta-Linguistic Form (@q): KEEP (TODO(@nhamilakis):  maybe should not ?)
    (substring_remover(seq="@q"), "@q"),
    # Phonetic Transcription (@u): Discard (TODO(@nhamilakis):  maybe should not ?)
    (word_remover(pattern=re.compile(r"\s[^ ]*@u\s?")), "@u"),
    # Letters (@l): KEEP
    (substring_remover(seq="@l"), "@l"),
    # Multi-letter (@k)
    (substring_remover(seq="@k"), "@k"),
    # Remove custom code Braunwald
    (substring_remover(seq="@z:sc"), "@z:sc"),
    # Excluded words (@x)
    (substring_remover(seq="@x"), "@x"),
    # Remove general form (1 instance)
    (substring_remover(seq="@g"), "@g"),
    # Remove accidental tags
    (substring_remover(seq="@m"), "@m"),
    # Remove Non-Comprehensible Speech
    (substring_remover(seq="xxx"), "xxx"),
    # Punctuation (That is surrounded by space)
    (substring_remover(seq=" . ", replace_with=" "), "."),
    (substring_remover(seq=" ? ", replace_with=" "), "?"),
    (substring_remover(seq=" ! ", replace_with=" "), "!"),
    (substring_remover(seq=" : ", replace_with=" "), ":"),
    (substring_remover(seq=" , ", replace_with=" "), ","),
    # Useless Characters (replace with space)
    (substring_remover(seq="↑", replace_with=" "), "↑"),
    (substring_remover(seq="↓", replace_with=" "), "↓"),
    (substring_remover(seq="≠", replace_with=" "), "≠"),
    (substring_remover(seq="‡", replace_with=" "), "‡"),
    (substring_remover(seq="+...", replace_with=" "), "+..."),
    (substring_remover(seq="+..?", replace_with=" "), "+..?"),
    (substring_remover(seq="+!?", replace_with=" "), "+!?"),
    (substring_remover(seq="+/.", replace_with=" "), "+/."),
    (substring_remover(seq="+/?", replace_with=" "), "+/?"),
    (substring_remover(seq="+//.", replace_with=" "), "+//."),
    (substring_remover(seq="+//?", replace_with=" "), "+//?"),
    (substring_remover(seq="+.", replace_with=" "), "+."),
    (substring_remover(seq="“", replace_with=" "), "“"),
    (substring_remover(seq="”", replace_with=" "), "”"),
    (substring_remover(seq='+"/.', replace_with=" "), '+"/.'),
    (substring_remover(seq='+".', replace_with=" "), '+".'),
    (substring_remover(seq='+"', replace_with=" "), '+"'),
    (substring_remover(seq="+^", replace_with=" "), "+^"),
    (substring_remover(seq="+,", replace_with=" "), "+,"),
    (substring_remover(seq="++", replace_with=" "), "++"),
    (substring_remover(seq="+<", replace_with=" "), "+<"),
    # Handle Compounds & Linkages (Remove <text> the '<' and '>' symbols but keep the text)
    (substring_remover(seq="<", replace_with=" "), "<"),
    (substring_remover(seq=">", replace_with=" "), ">"),
    # Phonological Fragments (&+): KEEP
    (substring_remover(seq="&+", replace_with=" "), "&+"),
    # Fillers (&-): KEEP
    (substring_remover(seq="&-", replace_with=" "), "&-"),
    # Actions (&=): Discard
    (word_remover(pattern=re.compile(r"&=[^\s]*")), "ACTIONS"),
    ## Inside word cleaning
    # TODO(@nhamilakis): handle NonCompletion of a Word (text(text)text)
    # TODO(@nhamilakis): handle Pause Between Syllables
    # TODO(@nhamilakis): handle Accronyms (F_B_I)
    # TODO(@nhamilakis): handle Word Stress ()
    # TODO(@nhamilakis): Handle Compounds (word_word_word)
]

cleaning_child_speech_rules: list[tuple[CLEANER_TYPE, str]] = []


class CleanerMeta:
    """Metadata storing class for cleaning job."""

    def __init__(self, file_id: str, speech_type: t.Literal["adult", "child"]) -> None:
        self.when = date.today()
        self.file_id = file_id
        self.speech_type = speech_type
        self.stats_dict: dict[str, int] = collections.defaultdict(lambda: 0)

    def add_stat(self, label: str, count: int) -> None:
        """Appends a statistic."""
        self.stats_dict[label] += count

    def export(self) -> dict:
        """Export stats as dictionary."""
        return {
            "when": self.when.isoformat(),
            "id": self.file_id,
            "speech_type": self.speech_type,
            "removed_stats": dict(self.stats_dict),
        }

    def dump(self, location: Path) -> None:
        """Dump stats in file."""
        with (location / f"{self.file_id}.stats.json").open("w") as fh:
            json.dump(self.export(), fh, indent=4)


def apply_cleaner_rules(file: Path, cleaner_rules: list[tuple[CLEANER_TYPE, str]], metadata: CleanerMeta) -> None:
    """Apply a set of cleaning rules to."""


class CHILDESCleaner:
    """Cleaner recipe for the CHILDES dataset."""

    def __init__(self, root_dir: Path, speech_type: t.Literal["adult", "child"]) -> None:
        self.speech_type = speech_type
        self.root_dir = root_dir
        if speech_type == "adult":
            self.rules = cleaning_adult_speech_rules
        elif speech_type == "child":
            self.rules = cleaning_child_speech_rules

    def extract_meta(self) -> None:
        """Extract meta annotations as per CHILDES::CHAT specification.

        Notes
        -----
            The CHILDES CHAT specification can be found here
                https://talkbank.org/manuals/CHAT.html

        """
        for file in self.root_dir.glob("*.raw"):
            metadata = CleanerMeta(file_id=file.stem, speech_type=self.speech_type)
            clean_file = []
            for line in file.read_text().splitlines():
                for rule, tag in self.rules:
                    clean_line, count = rule(line)
                    clean_file.append(clean_line)
                    metadata.add_stat(tag, count)

            # Dump clean content
            (file.with_suffix(".txt")).write_text("\n".join(clean_file))
