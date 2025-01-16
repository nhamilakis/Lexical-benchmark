import re
import typing as t

from lexical_benchmark import settings
from lexical_benchmark.datasets.utils import text_cleaning as txt


class TagCleaner(txt.TextActionFN):
    """A class helper for cleaning CHILDES tags."""

    def __init__(self, *, tag: str, label: str, keep: bool = True) -> None:
        super().__init__(label=label)
        self.clean_pattern = tag
        self.match_pattern = re.compile(f"\\b([^ ]+{tag})\\b")
        self.keep = keep

    def __call__(self, line: str) -> str:
        """Run clean Operation."""
        matches = self.match_pattern.findall(line)

        # Register Words
        for m in matches:
            self.add_word(self.label, m.replace(self.clean_pattern, ""))

        if self.keep:
            # Clean Matched items
            return line.replace(self.clean_pattern, "")

        # Return line cleanned of tagged words
        return self.match_pattern.sub(" ", line)


class AmpersandCleaner(txt.TextActionFN):
    """A class helper for cleaning CHILDES of & marked words."""

    def __init__(self, *, tag: str, label: str, keep: bool = True) -> None:
        super().__init__(label=label)
        self.clean_pattern = tag
        # TODO: check if this works with all &=
        self.match_pattern = re.compile(f"({tag}[^ ]+)")
        self.keep = keep

    def __call__(self, line: str) -> str:
        """Clean a line from given patterns."""
        matches = self.match_pattern.findall(line)

        if line.count(self.clean_pattern) != len(matches):
            self.add_error(
                self.label,
                msg=f"{matches=}, counting ({self.clean_pattern=}) found: "
                "{line.count(self.clean_pattern)=} in ({line})",
            )

        # Register Words
        for m in matches:
            self.add_word(self.label, m.replace(self.clean_pattern, ""))

        if self.keep:
            return line.replace(self.clean_pattern, "")
        return self.match_pattern.sub(" ", line)


class UndescoreReFormat(txt.TextActionFN):
    """A class helper to fix underscores in CHILDES.

    Underscores are used in two cases :

    1) Initials in CHILDES are formatted in the following way:
    - F_B_I -> F B I
    - A_M -> A M
    ...

    2) Groupings
    - I_would_like_a_ball -> I would like a ball
    ...
    """

    def __init__(self, speech_type: t.Literal["KCH", "OS", ""] = "") -> None:
        super().__init__(label=f"{speech_type}-undescore")
        self.pattern = re.compile(r"\b([A-Za-z]+(_[A-Za-z]+)+)\b")

    def __call__(self, line: str) -> str:
        """Clean line."""
        matches = self.pattern.findall(line)
        # Remove 2nd matches
        matches = [m[0] for m in matches]

        for m in matches:
            self.add_word(self.label, m)

        return line.replace("_", " ")


class InterpositionRemover(txt.TextActionFN):
    """A class helper to remove text interposition in CHILDES.

    Format: &*MOT:yes, &*FAT:mhm, ...
    """

    def __init__(self) -> None:
        super().__init__(label="&*")
        self.keep = bool
        self.pattern = re.compile(r"&\*[A-Z]+:[^\s]+")

    def __call__(self, line: str) -> str:
        """Match Interpositions and extract them from text."""
        matches = self.pattern.findall(line)
        clean_line = line

        for m in matches:
            self.add_word(self.label, m.replace("&*", ""))
            clean_line = clean_line.replace(m, " ")

        return clean_line


class WordCompletionRemover(txt.TextActionFN):
    """Cleaner for artificial word completion in CHILDES.

    Ex: (o)kay ==> the pronounced word was 'kay but the annotator added the (o) to signify the full word.

        - This rule needs to be applied after parenthesis annotation cleaning as to not confuse pause (.)
    for word completion.
        - This rule needs to be before "'" & "-" filtering.
    """

    def __init__(self) -> None:
        super().__init__(label="word-completion")
        self.match_pattern = re.compile(r"\w*\(\w+\)\w*")
        self.clean_pattern = re.compile(r"\(\w+\)")

    def __call__(self, line: str) -> str:
        """Clean line from word completion."""
        matches = self.match_pattern.findall(line)
        if matches:
            line = self.clean_pattern.sub("'", line)
            for m in matches:
                self.add_word(self.label, m)
        return line


# Remove Bracket  ([text]) Annotations
BRACKET_REMOVER = txt.PatternRemover(match=re.compile(r"\[([^\]]+)\]"), subst=" ", label="bracket-annotation")
# Remove Paranthesis Annotations
# BUG: this left some '(.)' in the dataset, all in the beggining of the phrase
PAREN_REMOVER = txt.PatternRemover(match=re.compile(r"\s\(([^()]*?)\)\s"), subst=" ", label="parenthesis-annotation")
TEXT_NORMALISATION = txt.TextNormalization()

PUNCTUATION_CLEANER = txt.MultiCharSeqRemover(
    "<",  # Groupings
    ">"  # Groupings
    # Common Punctiation
    ".",
    "?",
    "!",
    ",",
    # Useless Characters
    "+...",
    "+..?",
    "+!?",
    "+/.",
    "+/?",
    "+//.",
    "+//",
    "+//?",
    "+.",
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

XXX_NOISE_REMOVER = txt.CharSeqRemover(seq="xxx", label="XXX", count=True)
YYY_NOISE_REMOVER = txt.CharSeqRemover(seq="yyy", label="YYY", count=True)
WWW_NOISE_REMOVER = txt.CharSeqRemover(seq="www", label="WWW", count=True)


# BUG: match works but, counting fails as it looks for '&\\\\+' for some reason
PHONOLOGICAL_FRAGMENT_REMOVER = AmpersandCleaner(tag=r"&\+", label="&+", keep=False)
PHONOLOGICAL_FRAGMENT_CLEANER = AmpersandCleaner(tag=r"&\+", label="&+")

# BUG: fails as there are some loose '&~' with no word, they are properly cleaned though so false reporting.
NONWORD_REMOVER = AmpersandCleaner(tag=r"&~", label="&~", keep=False)
NONWORD_CLEANER = AmpersandCleaner(tag=r"&~", label="&~")

# Fillers (&-)
# BUG: match works ok, fails to count correctly the double &-uhh&-uh (39 instances):: no need to fix (probably)
FILLER_REMOVER = AmpersandCleaner(tag=r"&-", label="&-", keep=False)
FILLER_CLEANER = AmpersandCleaner(tag=r"&-", label="&-")

# Actions (&=): Discard
ACTION_REMOVER = AmpersandCleaner(tag=r"&=", label="&=", keep=False)

# Interposed speech (&*) : Discard
INTERPOSITION_REMOVER = InterpositionRemover()
UNDESCORE_CLEANER = UndescoreReFormat()

LEFTOVERCHARS_CLEANER = txt.MultiCharSeqRemover(
    "&",
    "+",
    ":",  # Word Stress
    "^",  # Word Stress2
    subst="",
    label="&:+",
)

# BUG(@nhamilakis): Some numbers fail to match (end of line ???)
NUMBER_CLEANER = txt.NumberFixer(keep_as_text=False)

WORD_COMPLETION_FIXER = WordCompletionRemover()

# TAG Cleaning
_adult_tag_removal: list[txt.CleanerFN] = [
    # Onomatopoeia (@o): KEEP
    TagCleaner(tag="@o", label="@o"),
    TagCleaner(tag="@p", label="@p"),
    # Babbling (@b): Discard
    TagCleaner(tag="@b", label="@b", keep=False),
    # Word-Play (@wp): Discard
    TagCleaner(tag="@wp", label="@wp)", keep=False),
    # Child Invented Form (@c): Discard
    TagCleaner(tag="@c", label="@c", keep=False),
    # Family Specific Form (@f): Discard
    TagCleaner(tag="@f", label="@f", keep=False),
    # Dialect Word (@d): KEEP
    TagCleaner(tag="@d", label="@d"),
    # Handle Second (or other) Language (@s:...) [Discard]
    *[TagCleaner(tag=f"{lang_tag}", label=f"({lang_tag})") for lang_tag in settings.CHILDES.EXTRA_LANGS],
    # Neologism (@n): KEEP
    TagCleaner(tag="@n", label="@n"),
    # Singing (@si): KEEP
    TagCleaner(tag="@si", label="@si"),
    # Interjection/interaction (@i): KEEP
    TagCleaner(tag="@i", label="@i"),
    # Test Words (@t) annotation : KEEP
    TagCleaner(tag="@t", label="@t"),
    # Meta-Linguistic Form (@q): KEEP
    TagCleaner(tag="@q", label="@q"),
    # Phonetic Transcription (@u): KEEP
    TagCleaner(tag="@u", label="@u"),
    # Letters (@l): KEEP
    TagCleaner(tag="@l", label="@l"),
    # Multi-letter (@k)
    TagCleaner(tag="@k", label="@k"),
    # Remove custom code Braunwald
    TagCleaner(tag="@z:sc", label="@z:sc"),
    # Excluded words (@x)
    TagCleaner(tag="@x", label="@x"),
    # Remove general form (1 instance)
    TagCleaner(tag="@g", label="@g"),
    # Remove accidental tags
    TagCleaner(tag="@m", label="@m"),
]

_child_tag_removal: list[txt.CleanerFN] = [
    # Onomatopoeia (@o): KEEP
    TagCleaner(tag="@o", label="@o"),
    TagCleaner(tag="@p", label="@p"),
    # Babbling (@b): Discard
    TagCleaner(tag="@b", label="@b"),
    # Word-Play (@wp): Discard
    TagCleaner(tag="@wp", label="@wp"),
    # Child Invented Form (@c): KEEP
    TagCleaner(tag="@c", label="@c"),
    # Family Specific Form (@f): KEEP
    TagCleaner(tag="@f", label="@f"),
    # Dialect Word (@d): KEEP
    TagCleaner(tag="@d", label="@d"),
    # Handle Second (or other) Language (@s:...) [Discard]
    *[TagCleaner(tag=f"{lang_tag}", label=f"({lang_tag})") for lang_tag in settings.CHILDES.EXTRA_LANGS],
    # Neologism (@n): KEEP
    TagCleaner(tag="@n", label="@n"),
    # Singing (@si): KEEP
    TagCleaner(tag="@si", label="@si"),
    # Interjection/interaction (@i): KEEP
    TagCleaner(tag="@i", label="@i"),
    # Test Words (@t) annotation : KEEP
    TagCleaner(tag="@t", label="@t"),
    # Meta-Linguistic Form (@q): KEEP
    TagCleaner(tag="@q", label="@q"),
    # Phonetic Transcription (@u): KEEP
    TagCleaner(tag="@u", label="@u"),
    # Letters (@l): KEEP
    TagCleaner(tag="@l", label="@l"),
    # Multi-letter (@k)
    TagCleaner(tag="@k", label="@k"),
    # Remove custom code Braunwald
    TagCleaner(tag="@z:sc", label="@z:sc"),
    # Excluded words (@x)
    TagCleaner(tag="@x", label="@x"),
    # Remove general form (1 instance)
    TagCleaner(tag="@g", label="@g"),
    # Remove accidental tags
    TagCleaner(tag="@m", label="@m"),
]

################################################################################################
# Rules per category of speech

cleaning_child_speech_rules: list[txt.CleanerFN] = [
    BRACKET_REMOVER,  # [...]
    PAREN_REMOVER,  # (..)
    TEXT_NORMALISATION,  # Fix accents & remove non ascii
    PUNCTUATION_CLEANER,  # Punctuation & single char annotations
    *_child_tag_removal,  # Clean or Remove tagged words/phrases
    # Noise Remover
    XXX_NOISE_REMOVER,
    YYY_NOISE_REMOVER,
    WWW_NOISE_REMOVER,
    # & Annotations are kept
    PHONOLOGICAL_FRAGMENT_CLEANER,
    NONWORD_CLEANER,
    FILLER_CLEANER,
    ACTION_REMOVER,
    INTERPOSITION_REMOVER,
    # Groupings
    UNDESCORE_CLEANER,
    # Symbols & Numbers
    NUMBER_CLEANER,
    WORD_COMPLETION_FIXER,
    LEFTOVERCHARS_CLEANER,
    txt.AZFilter(),
]

cleaning_adult_speech_rules: list[txt.CleanerFN] = [
    BRACKET_REMOVER,  # [...]
    PAREN_REMOVER,  # (..)
    TEXT_NORMALISATION,  # Fix accents & remove non ascii
    PUNCTUATION_CLEANER,  # Punctuation & single char annotations
    *_adult_tag_removal,  # Clean or Remove tagged words/phrases
    # Noise Remover
    XXX_NOISE_REMOVER,
    YYY_NOISE_REMOVER,
    WWW_NOISE_REMOVER,
    # & Annotations are kept
    PHONOLOGICAL_FRAGMENT_CLEANER,
    NONWORD_CLEANER,
    FILLER_CLEANER,
    ACTION_REMOVER,
    INTERPOSITION_REMOVER,
    # Groupings
    UNDESCORE_CLEANER,
    # Symbols & Numbers
    NUMBER_CLEANER,
    WORD_COMPLETION_FIXER,
    LEFTOVERCHARS_CLEANER,
    txt.AZFilter(),
]
