import json
import typing as t
from dataclasses import dataclass
from pathlib import Path

from lexical_benchmark import settings


@dataclass
class TurnTakeData:
    """Representation of turn-taking format."""

    adult_label: str
    adult: t.Literal["<EMPTY>"] | str  # noqa: PYI051
    child: t.Literal["<EMPTY>"] | str  # noqa: PYI051
    file_id: str = ""  # FileID is optional
    COLUMNS: t.ClassVar[tuple[str, ...]] = ("label", "adult_speech", "child_speech")

    def row(self) -> tuple[str, str, str]:
        """Row used to build turn-take data as csv."""
        return self.adult_label, self.adult, self.child


class TurnTakingBuilder:
    """Builder class for turn-taking sub-dataset."""

    @property
    def langs(self) -> tuple[str, ...]:
        """Languages Included in chiles."""
        return settings.CHILDES.ACCENTS

    def iter(self, lang: str) -> t.Iterable[Path]:
        """Iterator over files."""
        root = self.root_dir / lang / "txt"
        if not root.is_dir():
            raise ValueError(f"Lang {lang} not found in dataset")

        yield from root.glob("*.clean.json")

    def __init__(self, root_dir: Path = settings.PATH.clean_childes) -> None:
        self.root_dir = root_dir

    @staticmethod
    def merge_consecutive_speakers(dialog: list[list[str] | tuple[str, str]]) -> list[tuple[str, str]]:
        """Merge consecutive speakers in a dialog list."""
        merged_dialog: list[tuple[str, str]] = []

        if not dialog:  # If the input list is empty, return an empty list
            return merged_dialog

        # Initialize the first speaker's label and line
        current_label, current_lines = dialog[0]

        for label, line in dialog[1:]:
            if label == current_label:
                # If the speaker is the same, append the line to the current lines
                current_lines += " " + line
            else:
                # If the speaker changes, add the current speaker and their lines to the result
                merged_dialog.append((current_label, current_lines))
                # Update to the new speaker
                current_label, current_lines = label, line

        # Don't forget to add the last speaker's lines
        merged_dialog.append((current_label, current_lines))

        # Tag empty lines as UNINTELLIGIBLE
        marked_dialog = []
        for label, speech in merged_dialog:
            if speech == "":
                marked_dialog.append((label, "<UNINTELLIGIBLE>"))
            else:
                marked_dialog.append((label, speech))

        return marked_dialog

    @staticmethod
    def format_turn_taking(dialog: list[tuple[str, str]]) -> list[tuple[str, str, str]]:  # noqa: C901
        """Formating the dialog into turn-taking format."""
        formatted_dialog = []
        child_line = None
        adult_line = None
        adult_label = None

        for idx, (label, line) in enumerate(dialog):
            # We encounter child speech and have a previous adult speech registered
            if label == "CHI" and adult_label is not None:
                formatted_dialog.append((adult_label, adult_line, line))
                # reset registers
                adult_label, adult_line, child_line = None, None, None

            # We encounter child speech but no previous adult speech exists
            elif label == "CHI" and adult_label is None:
                if child_line is not None:
                    raise ValueError(f"illegal child consecutive speech item: {idx}")
                child_line = line

            # Non keyCHILD speech, with previously registered child speech
            elif label != "CHI" and child_line is not None:
                formatted_dialog.append((label, line, child_line))
                # reset registers
                adult_label, adult_line, child_line = None, None, None

            # Non keyCHILD speech, without previously registered child speech
            elif label != "CHI" and child_line is None:
                # Previous speaker is also Non keyCHIL
                if adult_label:
                    formatted_dialog.append((adult_label, adult_line, "<EMPTY>"))
                # update registers
                adult_label, adult_line = label, line

        # pushing odd leftovers on registries
        if child_line and adult_line and adult_label:
            formatted_dialog.append((adult_label, adult_line, child_line))
        elif child_line and adult_line is None:
            formatted_dialog.append(("-", "<EMPTY>", child_line))
        elif child_line is None and adult_line and adult_label:
            formatted_dialog.append((adult_label, adult_line, "<EMPTY>"))

        return formatted_dialog

    @classmethod
    def turn_taking_mk(cls, file: Path) -> list[TurnTakeData]:
        """Convert given file into the turn-taking format."""
        dialog = json.loads(file.read_bytes())
        merged_dialog = cls.merge_consecutive_speakers(dialog)
        turn_taking_dialog = cls.format_turn_taking(merged_dialog)
        # Return as turn-taking data
        return [TurnTakeData(*rows) for rows in turn_taking_dialog]
