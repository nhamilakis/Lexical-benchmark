import typing as t
from pathlib import Path

from rich import progress

from lexical_benchmark import settings
from lexical_benchmark.datasets.utils import lexicon
from lexical_benchmark.datasets.utils import text_cleaning as txt

from .data import STELLADatasetClean, STELLADatasetRaw


class StelaCleaner:
    """Cleaner recipe for the STELA Transcripts."""

    def __init__(self, src: Path = settings.PATH.raw_stela, target: Path = settings.PATH.clean_stela) -> None:
        self._progress_users = 0
        self._progress: progress.Progress | None = None
        # Navigation
        self.navigate = STELLADatasetRaw(root_dir=src)
        self.target = STELLADatasetClean(root_dir=target)

    @classmethod
    def get_ruleset(cls) -> list[txt.CleanerFN]:
        """Rules for cleaning Text."""
        return [
            txt.IllustrationRemoval(),  # Removes Illustration Tagging
            txt.URLRemover(),  # Remove URLs
            txt.SpecialCharacterTranscriptions(lang="EN", keep=True),
            txt.QuotationCleaner(),  # Clean quotes
            txt.TextNormalization(),  # Fix accents
            txt.NumberFixer(keep_as_text=True),  # Convert Numbers into text
            txt.RomanNumerals(),  # Remove Roman Numerals
            txt.AZFilter(),  # Removes any special character & punctuation
            txt.PrefixSuffixFixer(stem="'"),  # Remove prefix or suffix char(')
        ]

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
                refresh_per_second=10,
            )

        return self._progress

    def progress_stop(self) -> None:
        """Safely close progress."""
        self._progress_users -= 1
        if self._progress_users <= 0 and self._progress:
            self._progress.stop()

    def clean_txt(self, txt_dirty: list[str], *, ruleset: list[txt.CleanerFN]) -> list[str]:
        """Clean a the content of a txt file with the given ruleset."""
        return [txt.piped(f" {line} ", *ruleset) for line in txt_dirty]

    def cleanup_raw(self, *, show_progress: bool = False, compute_word_freqs: bool = False) -> None:
        """Clean transcriptions of Stela Dataset."""
        prg = self.progress(show_progress=show_progress)
        prg.start()
        clean_task = prg.add_task("Cleaning sections..", total=0)

        rules = self.get_ruleset()
        for idx, (lang, hour_split, section) in enumerate(self.navigate.iter_all()):
            prg.update(clean_task, total=idx)
            prg.update(clean_task, description=f"Cleaning {lang}/{hour_split}/{section}...")

            # Load source text
            _txt = self.navigate.transcription(lang, hour_split, section).read_text().splitlines()
            clean_txt = self.clean_txt(_txt, ruleset=rules)
            prg.update(clean_task, advance=0.5)

            # Write cleaned text
            self.navigate.clean_transcription(lang, hour_split, section).write_text("\n".join(clean_txt))

            # Save section logs
            logs = txt.WordLogger.dumps_logs()
            self.navigate.set_meta(logs, lang, hour_split, section)

            prg.update(clean_task, advance=0.25)

            # Compute words frequencies
            if compute_word_freqs:
                _ = self.navigate.words_freq(lang, hour_split, section)

            # progress update
            prg.update(clean_task, advance=0.25)

        # Finalize progres
        prg.update(clean_task, total=idx + 1, refresh=True)
        self.progress_stop()

    def _filter_words(self, language: str, hour_split: str, section: str, cleaner: lexicon.DictionairyCleaner) -> None:
        """Filter words of transcription."""
        bad_transcription_file = self.target.meta_dir(language, hour_split, section) / "bad.transcription.txt"
        transcription_file = self.target.transcription(language=language, hour_split=hour_split, section=section)
        source_text_file = self.navigate.clean_transcription(language=language, hour_split=hour_split, section=section)

        accepted_lines = []
        rejected_lines = []
        for line in source_text_file.read_text():
            accepted, rejected = cleaner(line)
            accepted_lines.append(accepted)
            rejected_lines.append(rejected)

        # Write clean text into transcription
        transcription_file.parent.mkdir(exist_ok=True, parents=True)
        self.target.transcription(language=language, hour_split=hour_split, section=section).write_text(
            "\n".join(accepted_lines)
        )

        # Write rejected lines
        bad_transcription_file.write_text("\n".join(rejected_lines))

    def mk_clean(
        self,
        word_cleaners: dict[str, lexicon.DictionairyCleaner],
        *,
        target: Path | None = None,
        compute_word_freqs: bool = False,
    ) -> None:
        """Make clean Stela dataset."""
        # Modify clean target dir
        if target:
            self.target.root_dir = target

        for _, (lang, hour_split, section) in enumerate(self.navigate.iter_all()):
            cleaner = word_cleaners.get(lang)
            if cleaner is None:
                print(f"No cleaner for {lang}, skipping")
                continue

            # Create clean, unclean subsets of transcriptions
            self._filter_words(lang, hour_split, section, cleaner=cleaner)

        # Compute word frequencies
        if compute_word_freqs:
            _ = self.target.clean_words_freq(lang, hour_split, section)
            _ = self.target.rejected_words_freq(lang, hour_split, section)

    def __enter__(self) -> t.Self:
        """Entering context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> t.Literal[False]:  # noqa: ANN001
        """Exit Context."""
        # Release progress object
        if self._progress:
            self._progress.stop()

        return False  # Propagate exceptions, if any
