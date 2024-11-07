import typing as t
import warnings
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
        accepted_lines = []
        rejected_lines = []
        for line in txt:
            accepted, rejected = cleaner(line)
            accepted_lines.append(accepted)
            rejected_lines.append(rejected)
        return accepted_lines, rejected_lines

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

            _txt = source_file.safe_readlines()
            if _txt is None:
                warnings.warn(f"File {source_file} does not exist !!", category=UserWarning, stacklevel=1)
                continue
            clean_txt = cls.clean_txt(_txt, ruleset=ruleset)

            # Write cleaned text
            target_file.safe_write_text("\n".join(clean_txt))

            # Save section logs
            logs = text_cleaning.WordLogger.dumps_logs()
            if save_logs:
                logfile.dump_json(logs)

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

            _txt = source_file.safe_readlines()
            if _txt is None:
                warnings.warn(f"File {source_file} does not exist !!", category=UserWarning, stacklevel=1)
                continue
            accepted_text, rejected_text = cls.word_validator(_txt, cleaner=cleaner)

            # Write cleaned @ rejected text into corresponding files
            clean_target.safe_write_text("\n".join(accepted_text))  # type: ignore[attr-defined] # ducktyping
            reject_target.safe_write_text("\n".join(rejected_text))  # type: ignore[attr-defined] # ducktyping
