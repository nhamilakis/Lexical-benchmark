import typing as t
from pathlib import Path

from . import lexicon, text_cleaning


class DatasetCleaner:
    """Cleaner recipe for the STELA Transcripts."""

    @staticmethod
    def clean_txt(txt_dirty: list[str], *, ruleset: list[text_cleaning.CleanerFN]) -> list[str]:
        """Clean a the content of a txt file with the given ruleset."""
        return [text_cleaning.piped(f" {line} ", *ruleset) for line in txt_dirty]

    @staticmethod
    def word_validator(txt: list[str], *, cleaner: lexicon.DictionairyCleaner) -> tuple[list[str], list[str]]:
        """Validate words from a text by passing them through a dictionairy.

        Returns
        -------
            Two lists containing accepted & rejected lines

        """
        return zip(*[cleaner(line) for line in txt], strict=True)  # type: ignore[return-value]

    @classmethod
    def cleanup_files(
        cls,
        *,
        filemap: t.Sequence[tuple[Path, Path, Path]],
        ruleset: list[text_cleaning.CleanerFN],
        save_logs: bool = True,
    ) -> None:
        """Clean files using given ruleset.

        Args:
        ----
            filemap: Path objects with the files to clean in a list, paired with their correspondint target
                     for cleaned text.
            ruleset: the list of rules to use for cleaning.
            save_logs: deactivate log saving

        """
        for _, (source_file, target_file, logfile) in enumerate(filemap):
            # Load source text

            _txt = source_file.read_text().splitlines()
            clean_txt = cls.clean_txt(_txt, ruleset=ruleset)

            # Write cleaned text
            target_file.safe_write_text("\n".join(clean_txt))  # type: ignore[attr-defined] # ducktyping

            # Save section logs
            logs = text_cleaning.WordLogger.dumps_logs()
            if save_logs:
                logfile.dump_json(logs)  # type: ignore[attr-defined] # ducktyping

    @classmethod
    def word_validate_files(
        cls, *, filemap: t.Sequence[tuple[Path, Path, Path]], cleaner: lexicon.DictionairyCleaner
    ) -> None:
        """Filter given files through a Dictionairy validator.

        Args:
        ----
            filemap: Path objects with the files to clean in a list, paired with their correspondint target
                     for accepted & rejected text.
            cleaner: the dictionairy to use for validation filtering.

        """
        for _, (source_file, clean_target, reject_target) in enumerate(filemap):
            # Load source text

            _txt = source_file.read_text().splitlines()
            accepted_text, rejected_text = cls.word_validator(_txt, cleaner=cleaner)

            # Write cleaned @ rejected text into corresponding files
            clean_target.safe_write_text("\n".join(accepted_text))  # type: ignore[attr-defined] # ducktyping
            reject_target.safe_write_text("\n".join(rejected_text))  # type: ignore[attr-defined] # ducktyping
