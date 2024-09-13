import typing as t
from pathlib import Path

from rich import progress

from lexical_benchmark import settings
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
            txt.TextNormalization(),  # Fix accents
            txt.NumberFixer(keep_as_text=True),  # Convert Numbers into text
            txt.RomanNumerals(),  # Remove Roman Numerals
            txt.AZFilter(),  # Removes any special character & punctuation
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

    def mk_clean(self, *, show_progress: bool = False) -> None:
        """Clean transcriptions of Stela Dataset."""
        prg = self.progress(show_progress=show_progress)
        prg.start()
        clean_task = prg.add_task("Cleaning sections..", total=0)

        rules = self.get_ruleset()
        for idx, (lang, hour_split, section) in enumerate(self.navigate.iter_all()):
            prg.update(clean_task, total=idx)
            prg.update(clean_task, description=f"Cleaning {lang}/{hour_split}/{section}...")

            _txt = self.navigate.get_transcription(lang, hour_split, section)
            clean_txt = self.clean_txt(_txt, ruleset=rules)
            prg.update(clean_task, advance=0.5)

            self.target.set_transcription(clean_txt, lang, hour_split, section)

            # Save section logs
            logs = txt.WordLogger.dumps_logs()
            self.target.set_meta(logs, lang, hour_split, section)

            prg.update(clean_task, advance=0.25)

            # Compute all words frequencies
            _ = self.target.all_words_freq(lang, hour_split, section)

            # progress update
            prg.update(clean_task, advance=0.25)

        # Finalize progres
        prg.update(clean_task, total=idx + 1, refresh=True)
        self.progress_stop()

    def __enter__(self) -> t.Self:
        """Entering context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> t.Literal[False]:  # noqa: ANN001
        """Exit Context."""
        # Release progress object
        if self._progress:
            self._progress.stop()

        return False  # Propagate exceptions, if any
