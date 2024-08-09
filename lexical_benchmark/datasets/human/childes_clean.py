import collections
import functools
import json
import re
import typing as t
from datetime import date
from pathlib import Path

CLEANER_TYPE = t.Callable[[str], tuple[str, int]]


def pattern_remove(line: str, *, pattern: t.Pattern[str], replace_with: str = "") -> tuple[str, int]:
    """Removes given pattern from given string."""
    return pattern.sub(replace_with, line), len(pattern.findall(line))


def substring_remove(line: str, *, seq: str, replace_with: str = "") -> tuple[str, int]:
    """Replace sequence of characters from given string."""
    return line.replace(seq, replace_with), line.count(seq)


cleaning_adult_speech_rules: list[tuple[CLEANER_TYPE, str]] = [
    # Extracts all items between brackets from each line in a text file.
    (functools.partial(pattern_remove, pattern=re.compile(r"\[([^\]]+)\]")), "bracket-annotation"),
    # Remove Paranthesis
    (functools.partial(pattern_remove, pattern=re.compile(r"\s\(([^()]*?)\)\s")), "parenthesis-annotation"),
    # TODO(@nhamilakis): handle onomatopoeia (@o)
    # TODO(@nhamilakis): handle Phonological consistent forms (@p)
    # TODO(@nhamilakis): handle Babbling (@b)
    # TODO(@nhamilakis): handle Word-Play (@wp)
    # TODO(@nhamilakis): handle Child Invented Form (@c)
    # TODO(@nhamilakis): handle Family Specific Form (@f)
    # TODO(@nhamilakis): handle Dialect Word (@d)
    # TODO(@nhamilakis): handle Second (or other) Language (@s:...)
    # TODO(@nhamilakis): handle Neologism (@n)
    # TODO(@nhamilakis): handle Singing (@si)
    # TODO(@nhamilakis): handle interjection/interaction (@i)
    # Remove Test Words (@t) annotation
    (functools.partial(substring_remove, seq="@t"), "@t"),
    # TODO(@nhamilakis): handle Meta-Linguistic Form (@q)
    # TODO(@nhamilakis): handle Phonetic Transcription (@u)
    # TODO(@nhamilakis): handle Letters (@l)
    # TODO(@nhamilakis): handle Multi-letter (@k)
    # Remove custom code Braunwald
    (functools.partial(substring_remove, seq="@z:sc"), "@z:sc"),
    # Excluded words (@x)
    (functools.partial(substring_remove, seq="@x"), "@x"),
    # Remove general form (1 instance)
    (functools.partial(substring_remove, seq="@g"), "@g"),
    # Remove accidental tags
    (functools.partial(substring_remove, seq="@m"), "@m"),
    # Remove Non-Comprehensible Speech
    (functools.partial(substring_remove, seq="xxx"), "xxx"),
    # Punctuation (That is surrounded by space)
    (functools.partial(substring_remove, seq=" . ", replace_with=" "), "."),
    (functools.partial(substring_remove, seq=" ? ", replace_with=" "), "?"),
    (functools.partial(substring_remove, seq=" ! ", replace_with=" "), "!"),
    (functools.partial(substring_remove, seq=" : ", replace_with=" "), ":"),
    (functools.partial(substring_remove, seq=" , ", replace_with=" "), ","),
    # Useless Characters (replace with space)
    (functools.partial(substring_remove, seq="↑", replace_with=" "), "↑"),
    (functools.partial(substring_remove, seq="↓", replace_with=" "), "↓"),
    (functools.partial(substring_remove, seq="≠", replace_with=" "), "≠"),
    (functools.partial(substring_remove, seq="‡", replace_with=" "), "‡"),
    (functools.partial(substring_remove, seq="+...", replace_with=" "), "+..."),
    (functools.partial(substring_remove, seq="+..?", replace_with=" "), "+..?"),
    (functools.partial(substring_remove, seq="+!?", replace_with=" "), "+!?"),
    (functools.partial(substring_remove, seq="+/.", replace_with=" "), "+/."),
    (functools.partial(substring_remove, seq="+/?", replace_with=" "), "+/?"),
    (functools.partial(substring_remove, seq="+//.", replace_with=" "), "+//."),
    (functools.partial(substring_remove, seq="+//?", replace_with=" "), "+//?"),
    (functools.partial(substring_remove, seq="+.", replace_with=" "), "+."),
    (functools.partial(substring_remove, seq="“", replace_with=" "), "“"),
    (functools.partial(substring_remove, seq="”", replace_with=" "), "”"),
    (functools.partial(substring_remove, seq='+"/.', replace_with=" "), '+"/.'),
    (functools.partial(substring_remove, seq='+".', replace_with=" "), '+".'),
    (functools.partial(substring_remove, seq='+"', replace_with=" "), '+"'),
    (functools.partial(substring_remove, seq="+^", replace_with=" "), "+^"),
    (functools.partial(substring_remove, seq="+,", replace_with=" "), "+,"),
    (functools.partial(substring_remove, seq="++", replace_with=" "), "++"),
    (functools.partial(substring_remove, seq="+<", replace_with=" "), "+<"),
    # TODO(@nhamilakis): handle Phonological Fragments (&+)
    # TODO(@nhamilakis): handle Fillers (&-)
    # TODO(@nhamilakis): handle Remove Actions (&=)
    ## Inside word cleaning
    # TODO(@nhamilakis): handle NonCompletion of a Word (text(text)text)
    # TODO(@nhamilakis): handle Pause Between Syllables
    # TODO(@nhamilakis): handle Accronyms (F_B_I)
    # TODO(@nhamilakis): handle Compounds & Linkages
    # TODO(@nhamilakis): handle Word Stress ()
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
