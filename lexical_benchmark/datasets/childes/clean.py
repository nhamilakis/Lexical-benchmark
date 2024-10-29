import json
import re
import typing as t
from pathlib import Path

from rich import progress

from lexical_benchmark import settings, utils
from lexical_benchmark.datasets.utils import text_cleaning as txt

from .data import SPEECH_TYPE, RawCHILDESFiles, TXTItem


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
        # TODO check if this works with all &=
        self.match_pattern = re.compile(f"({tag}[^ ]+)")
        self.keep = keep

    def __call__(self, line: str) -> str:
        """Clean a line from given patterns."""
        matches = self.match_pattern.findall(line)

        if line.count(self.clean_pattern) != len(matches):
            self.add_error(
                self.label,
                msg=f"{matches=}, counting ({self.clean_pattern=}) found: {line.count(self.clean_pattern)=} in ({line})",
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


# BUG(@nhamilakis): match works but, counting fails as it looks for '&\\\\+' for some reason
PHONOLOGICAL_FRAGMENT_REMOVER = AmpersandCleaner(tag=r"&\+", label="&+", keep=False)
PHONOLOGICAL_FRAGMENT_CLEANER = AmpersandCleaner(tag=r"&\+", label="&+")

# BUG(@nhamilakis): fails as there are some loose '&~' with no word, they are properly cleaned though so false reporting.
NONWORD_REMOVER = AmpersandCleaner(tag=r"&~", label="&~", keep=False)
NONWORD_CLEANER = AmpersandCleaner(tag=r"&~", label="&~")

# Fillers (&-)
# BUG(@nhamilakis): match works ok, fails to count correctly the double &-uhh&-uh (39 instances):: no need to fix (probably)
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


class CHILDESCleaner:
    """Cleaner recipe for the CHILDES dataset."""

    @classmethod
    def get_ruleset(cls, speech_type: SPEECH_TYPE) -> list[txt.CleanerFN]:
        """Build Rules for speech-types."""
        if speech_type == "child":
            return cleaning_child_speech_rules

        if speech_type == "adult":
            return cleaning_adult_speech_rules

        raise ValueError(f"Unknown speech-type: {speech_type}")

    def __init__(self) -> None:
        self.file_nav = RawCHILDESFiles()
        self._progress: progress.Progress | None = None
        self._progress_users = 0

    def progress(self, *, show_progress: bool = False) -> progress.Progress:
        """Load or fetch progress item."""
        self._progress_users += 1

        if self._progress:
            self._progress.disable = not show_progress
        else:
            self._progress = progress.Progress(
                progress.TextColumn("[progress.description]{task.description}"),
                progress.BarColumn(),
                progress.MofNCompleteColumn(),
                progress.TimeElapsedColumn(),
                expand=True,
                disable=not show_progress,
                auto_refresh=True,
            )

        return self._progress

    def progress_stop(self) -> None:
        """Safely close progress."""
        self._progress_users -= 1
        if self._progress_users <= 0 and self._progress:
            self._progress.stop()

    def clean_files(
        self,
        target: Path,
        files_iter: t.Iterable[TXTItem | Path],
        ruleset: list[txt.CleanerFN],
        *,
        show_progress: bool = False,
    ) -> None:
        """Clean a list of CHILDES speech-files."""
        prg = self.progress(show_progress=show_progress)
        prg.start()
        file_task = prg.add_task("Cleaning files..", total=0)

        for idx, item in enumerate(files_iter):
            prg.update(file_task, total=idx)

            file = item.file if isinstance(item, TXTItem) else item
            prg.update(file_task, description=f"Cleaning {file.stem}...")

            # Pass lines through cleaning pipeline
            clean_lines = [txt.piped(f" {line} ", *ruleset) for line in file.read_text().splitlines()]
            # Write results in clean dataset
            (target / file.with_suffix(".txt").name).write_text("\n".join(clean_lines))
            # Dump & reset logs
            txt.WordLogger.dump_logs(file=file.with_suffix(".meta.json"))

            # Advance task
            prg.update(file_task, advance=1)

        prg.update(file_task, total=idx + 1, refresh=True)
        self.progress_stop()

    def clean_txt_files(
        self,
        target: Path,
        files_iter: t.Iterable[TXTItem | Path],
        child_ruleset: list[txt.CleanerFN],
        adult_ruleset: list[txt.CleanerFN],
    ) -> None:
        """Clean a list of json files containing CHILDES speech."""

        def _clean_line(label: str, line: str) -> tuple[str, str]:
            """Internal cleans line function."""
            if "CHI" in label:
                return label, txt.piped(f" {line} ", *child_ruleset)
            return label, txt.piped(f" {line} ", *adult_ruleset)

        for _, item in enumerate(files_iter):
            file = item.file if isinstance(item, TXTItem) else item

            # Parse & clean file
            as_json = json.loads(file.read_bytes())
            clean_lines = [_clean_line(label, line) for label, line in as_json]

            # Dump clean text
            as_txt = json.dumps(clean_lines, indent=4, default=utils.default_json_encoder)
            (target / file.with_suffix(".json").name).write_text(as_txt)

            # Dump & reset logs
            txt.WordLogger.dump_logs(file=file.with_suffix(".meta.json"))

    def mk_clean(self, target: Path = settings.PATH.clean_childes, *, show_progress: bool = False) -> None:
        """Clean all of CHILDES Dataset, and create clean-version."""
        prg = self.progress(show_progress=show_progress)
        prg.start()

        langs = self.file_nav.langs
        speech_types = t.get_args(SPEECH_TYPE)

        for lang in prg.track(langs, description="Cleaning Childes languages..."):
            for speech in prg.track(speech_types, description=f"Cleaning targets in {lang}.."):
                location = target / lang / speech
                location.mkdir(exist_ok=True, parents=True)

                # Clean files in set
                self.clean_files(
                    target=location,
                    files_iter=self.file_nav.iter(lang, speech),
                    ruleset=self.get_ruleset(speech),
                    show_progress=show_progress,
                )

        self.progress_stop()

    def mk_clean_txt(self, target: Path = settings.PATH.clean_childes) -> None:
        """Clean the txt section of the dataset."""
        for lang in self.file_nav.langs:
            location = target / lang / "txt"
            location.mkdir(exist_ok=True, parents=True)

            self.clean_txt_files(
                target=location,
                files_iter=(self.file_nav.root_dir / lang / "txt").glob("*.json"),
                child_ruleset=self.get_ruleset("child"),
                adult_ruleset=self.get_ruleset("adult"),
            )
