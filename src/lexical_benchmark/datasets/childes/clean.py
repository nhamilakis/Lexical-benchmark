import json
import typing as t
from pathlib import Path

from rich import progress

from lexical_benchmark import settings, utils
from lexical_benchmark.datasets.utils import text_cleaning as txt

from .cleanup_rules import cleaning_adult_speech_rules, cleaning_child_speech_rules
from .data import SPEECH_TYPE, RawCHILDESFiles, TXTItem


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
