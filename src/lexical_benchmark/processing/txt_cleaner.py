import typing as t
import warnings
from pathlib import Path

from lexical_benchmark.text_lib import lexicon, text_cleaners


class DatasetCleaner:
    """Cleaner recipe for the STELA Transcripts."""

    @staticmethod
    def dump_logs() -> dict:
        """Export cleaning logs."""
        return text_cleaners.WordLogger.dumps_logs()

    @staticmethod
    def clean_txt(txt_dirty: list[str], *, ruleset: list[text_cleaners.CleanerFN]) -> list[str]:
        """Clean a the content of a txt file with the given ruleset."""
        return [text_cleaners.piped(f" {line} ", *ruleset) for line in txt_dirty]

    @staticmethod
    def line_filter(txt: list[str], *, line_filter_fn: lexicon.DictionairyCleaner) -> tuple[list[str], list[str]]:
        """Validate words from a text by passing them through a dictionairy.

        Returns
        -------
            Two lists containing accepted & rejected lines

        """
        accepted_lines = []
        rejected_lines = []
        for line in txt:
            accepted, rejected = line_filter_fn(line)
            accepted_lines.append(accepted)
            rejected_lines.append(rejected)
        return accepted_lines, rejected_lines

    @classmethod
    def cleanup_files(
        cls,
        *,
        filemap: t.Sequence[tuple[Path, Path, Path]] | t.Iterable[tuple[Path, Path, Path]],
        ruleset: list[text_cleaners.CleanerFN],
        save_logs: bool = True,
    ) -> None:
        """Clean files using given ruleset."""
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
            logs = text_cleaners.WordLogger.dumps_logs()
            if save_logs and logfile:
                logfile.dump_json(logs)

    @classmethod
    def filter_files(
        cls,
        *,
        filemap: t.Sequence[tuple[Path, Path, Path]] | t.Iterable[tuple[Path, Path, Path]],
        lex: lexicon.DictionairyCleaner,
    ) -> None:
        """Filter given files through a Dictionairy validator."""
        for _, (source_file, clean_target, reject_target) in enumerate(filemap):
            # Load source text

            _txt = source_file.safe_readlines()
            if _txt is None:
                warnings.warn(f"File {source_file} does not exist !!", category=UserWarning, stacklevel=1)
                continue
            accepted_text, rejected_text = cls.line_filter(_txt, line_filter_fn=lex)

            # Write cleaned @ rejected text into corresponding files
            clean_target.safe_write_text("\n".join(accepted_text))  # type: ignore[attr-defined] # ducktyping
            reject_target.safe_write_text("\n".join(rejected_text))  # type: ignore[attr-defined] # ducktyping
