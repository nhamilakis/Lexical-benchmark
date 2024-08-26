import collections
import functools
import json
import re
import typing as t
from datetime import date
from pathlib import Path


class WordLogger:
    """Class that allows logging of words."""

    _LOGS: t.ClassVar[dict[str, list[str]]] = collections.defaultdict(list)
    _COUNT_LOGS: t.ClassVar[dict[str, int]] = collections.defaultdict(lambda: 0)

    @classmethod
    def add_word(cls, label: str, word: str) -> None:
        """Save a clean word to the Log."""
        cls._LOGS[label].append(word)

    @classmethod
    def update_log(cls, label: str) -> None:
        """Add one to the count of an action."""
        cls._COUNT_LOGS[label] += 1

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


class CleanerFN(t.Protocol):
    """Protocol for clean function-class."""

    def __call__(self, line: str) -> str:
        """Run clean."""


class PatternRemover(WordLogger):
    """A class to help remove patterns."""

    def __init__(self, *, match: t.Pattern[str], label: str, subst: str = "") -> None:
        self.pattern = match
        self.label = label
        self.subst = subst

    def __call__(self, line: str) -> str:
        """Run clean Operation."""
        matches = self.pattern.findall(line)

        # Register Words
        for m in matches:
            self.add_word(self.label, m)

        # Return line without matches
        return self.pattern.sub(self.subst, line)


class TagCleaner(WordLogger):
    """A class helper for cleaning CHILDES tags."""

    def __init__(self, *, tag: str, label: str, keep: bool = True) -> None:
        self.clean_pattern = tag
        self.match_pattern = re.compile(f"\\b[^ ]+{tag}\\b")
        self.label = label
        self.keep = keep

    def __call__(self, line: str) -> str:
        """Run clean Operation."""
        matches = self.match_pattern.findall(line)

        # Register Words
        for m in matches:
            self.add_word(self.label, m.replace(self.clean_pattern, ""))

        if self.keep:
            # Clean Matched items
            clean_line = line
            for m in matches:
                clean_line = line.replace(m, "")
            return clean_line

        # Return line cleanned of tagged words
        return line.replace(self.clean_pattern, "")


class CharSeqRemover(WordLogger):
    """A class to help removing chars sequences from text."""

    def __init__(self, *chars: str, label: str, subst: str = "") -> None:
        self.chars = chars
        self.label = label
        self.subst = subst

    def __call__(self, line: str) -> str:
        """Clean given line from specific chars."""
        clean = line
        for c in self.chars:
            clean = line.replace(c, self.subst)
        return clean


# def pattern_remover(*, patern: t.Pattern[str], replace_with: str = "") -> CLEANER_TYPE:
#     """Removes given pattern from given string."""

#     def _internal_fn(line: str, *, _pattern: t.Pattern[str], _replace_with: str = replace_with) -> tuple[str, int]:
#         return _pattern.sub(_replace_with, line), len(_pattern.findall(line))

#     # Return as a partial rule
#     return functools.partial(_internal_fn, _pattern=patern, _replace_with=replace_with)


# def substring_remover(*, seq: str, replace_with: str = "") -> CLEANER_TYPE:
#     """Replace sequence of characters from given string."""

#     def _internal_fn(line: str, _seq: str, _replace_with: str = "") -> tuple[str, int]:
#         return line.replace(_seq, _replace_with), line.count(_seq)

#     return functools.partial(_internal_fn, _seq=seq, _replace_with=replace_with)


# def word_remover(*, pattern: t.Pattern[str]) -> CLEANER_TYPE:
#     """Remove a word from the coprus."""
#     return pattern_remover(patern=pattern, replace_with=" ")

_common_precleaning_rules: list[CleanerFN] = [
    # Remove Bracket  ([text]) Annotations
    PatternRemover(match=re.compile(r"\[([^\]]+)\]"), label="bracket-annotation"),
    # Remove Paranthesis Annotations
    PatternRemover(match=re.compile(r"\s\(([^()]*?)\)\s"), label="parenthesis-annotation"),
    CharSeqRemover("<", ">", label="<>"),
]

_punctuation_cleaner: CleanerFN = CharSeqRemover(
    # Common Punctiation
    ".",
    "?",
    "!",
    ",",
    # Useless Characters
    "↑",
    "↓",
    "≠",
    "‡",
    "+...",
    "+..?",
    "+!?",
    "+/.",
    "+/?",
    "+//.",
    "+//?",
    "+.",
    "“",
    "”",
    '+"/.',
    '+".',
    '+"',
    "+^",
    "+,",
    "++",
    "+<",
    label="common-punctuation",
    subst=" ",
)


_adult_tag_removal: list[CleanerFN] = [
    # Onomatopoeia (@o): KEEP
    TagCleaner(tag="@o", label="Adult-Onomatopoia(@o)"),
    TagCleaner(tag="@p", label="Adult-PhonologicalConsistentForm(@p)"),
    # Babbling (@b): Discard
    TagCleaner(tag="@b", label="Adult-Babling(@b)", keep=False),
    # Word-Play (@wp): Discard
    TagCleaner(tag="@wp", label="Adult-WordPlay(@wp)", keep=False),
    # Child Invented Form (@c): Discard
    TagCleaner(tag="@wp", label="Adult-ChildInventedForm(@c)", keep=False),
    # Family Specific Form (@f): Discard
    TagCleaner(tag="@f", label="Adult-FamilySpecificForm(@f)", keep=False),
    # Dialect Word (@d): KEEP
    TagCleaner(tag="@d", label="Adult-Dialect(@d)"),
    # TODO(@nhamilakis): handle Second (or other) Language (@s:...) [Discard]
    # Neologism (@n): KEEP
    TagCleaner(tag="@n", label="Adult-Neologism(@n)"),
    # Singing (@si): KEEP
    TagCleaner(tag="@si", label="Adult-Singing(@si)"),
    # Interjection/interaction (@i): KEEP
    TagCleaner(tag="@i", label="Adult-Interjection(@i)"),
    # Test Words (@t) annotation : KEEP
    TagCleaner(tag="@t", label="Adult-TestWords(@t)"),
    # Meta-Linguistic Form (@q): KEEP
    TagCleaner(tag="@q", label="Adult-MetaLForm(@q)"),
    # Phonetic Transcription (@u): KEEP
    TagCleaner(tag="@u", label="Adult-Phonetic(@u)"),
    # Letters (@l): KEEP
    TagCleaner(tag="@l", label="Adult-Letters(@l)"),
    # Multi-letter (@k)
    TagCleaner(tag="@k", label="Adult-Letters(@k)"),
    # Remove custom code Braunwald
    TagCleaner(tag="@z:sc", label="Adult-BraunwaldCode(@z:sc)"),
    # Excluded words (@x)
    TagCleaner(tag="@x", label="Adult-ExcludedWords(@x)"),
    # Remove general form (1 instance)
    TagCleaner(tag="@g", label="Adult-GeneralForm(@g)"),
    # Remove accidental tags
    TagCleaner(tag="@m", label="Adult-ErrorTags(@m)"),
]

_child_tag_removal: list[CleanerFN] = [
    # Onomatopoeia (@o): KEEP
    TagCleaner(tag="@o", label="CHILD-Onomatopoia(@o)"),
    TagCleaner(tag="@p", label="CHILD-PhonologicalConsistentForm(@p)"),
    # Babbling (@b): Discard
    TagCleaner(tag="@b", label="CHILD-Babling(@b)"),
    # Word-Play (@wp): Discard
    TagCleaner(tag="@wp", label="CHILD-WordPlay(@wp)"),
    # Child Invented Form (@c): Discard
    TagCleaner(tag="@wp", label="CHILD-ChildInventedForm(@c)"),
    # Family Specific Form (@f): Discard
    TagCleaner(tag="@f", label="CHILD-FamilySpecificForm(@f)"),
    # Dialect Word (@d): KEEP
    TagCleaner(tag="@d", label="CHILD-Dialect(@d)"),
    # TODO(@nhamilakis): handle Second (or other) Language (@s:...) [Discard]
    # Neologism (@n): KEEP
    TagCleaner(tag="@n", label="CHILD-Neologism(@n)"),
    # Singing (@si): KEEP
    TagCleaner(tag="@si", label="CHILD-Singing(@si)"),
    # Interjection/interaction (@i): KEEP
    TagCleaner(tag="@i", label="CHILD-Interjection(@i)"),
    # Test Words (@t) annotation : KEEP
    TagCleaner(tag="@t", label="CHILD-TestWords(@t)"),
    # Meta-Linguistic Form (@q): KEEP
    TagCleaner(tag="@q", label="CHILD-MetaLForm(@q)"),
    # Phonetic Transcription (@u): KEEP
    TagCleaner(tag="@u", label="CHILD-Phonetic(@u)"),
    # Letters (@l): KEEP
    TagCleaner(tag="@l", label="CHILD-Letters(@l)"),
    # Multi-letter (@k)
    TagCleaner(tag="@k", label="CHILD-Letters(@k)"),
    # Remove custom code Braunwald
    TagCleaner(tag="@z:sc", label="CHILD-BraunwaldCode(@z:sc)"),
    # Excluded words (@x)
    TagCleaner(tag="@x", label="CHILD-ExcludedWords(@x)"),
    # Remove general form (1 instance)
    TagCleaner(tag="@g", label="CHILD-GeneralForm(@g)"),
    # Remove accidental tags
    TagCleaner(tag="@m", label="CHILD-ErrorTags(@m)"),
]

# cleaning_adult_speech_rules: list[CleanerFN] = [
#     # Remove Non-Comprehensible Speech
#     (substring_remover(seq="xxx"), "xxx"),
#     # Phonological Fragments (&+): KEEP
#     (substring_remover(seq="&+", replace_with=" "), "&+"),
#     # Fillers (&-): KEEP
#     (substring_remover(seq="&-", replace_with=" "), "&-"),
#     # Actions (&=): Discard
#     (word_remover(pattern=re.compile(r"&=[^\s]*")), "ACTIONS"),
#     ## Inside word cleaning
#     # TODO(@nhamilakis): handle NonCompletion of a Word (text(text)text)
#     # TODO(@nhamilakis): handle Pause Between Syllables
#     # TODO(@nhamilakis): handle Accronyms (F_B_I)
#     # TODO(@nhamilakis): handle Word Stress ()
#     # TODO(@nhamilakis): Handle Compounds (word_word_word)
# ]

cleaning_child_speech_rules: list[CleanerFN] = [
    *_common_precleaning_rules,
    _punctuation_cleaner,
    *_child_tag_removal,
    # TODO(@nhamilakis): Add post tag cleanup
]

cleaning_adult_speech_rules: list[CleanerFN] = [
    *_common_precleaning_rules,
    _punctuation_cleaner,
    *_adult_tag_removal,
    # TODO(@nhamilakis): Add post tag cleanup
]


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
