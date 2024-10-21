import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.progress import track

from lexical_benchmark import settings, utils
from lexical_benchmark.datasets import parsing


def parse_childes_age(age: str) -> float:
    """Parse string based age and convert it into months.

    The age of the childs is formated in the folowing way :

                "Y;MM.MM"

    where Y is the number of years, MM is the number of months.
    The year is an int and the months are formatted as float (to include days).

    Returns
    -------
        float: as the number of months

    """
    age = str(age)
    if ";" not in age:
        try:
            return float(age)
        except ValueError:
            return np.nan

    year, _, month = age.partition(";")
    try:
        return (int(year) * 12) + float(month)
    except ValueError:
        return np.nan


class CHILDESPreparation:
    """Class used to pre-format the CHILDES dataset.

    Output Dataset Structure:
        EN_UK/
            metadata.csv
            child_speech/
                *.raw
            adult_speech/
                *.raw
        EN_NA/
            metadata.csv
            child_speech/
                *.raw
            adult_speech/
                *.raw
    """

    def __init__(self) -> None:
        self.dataset: dict[str, list[tuple[str, Path]]] = {}

    def load_dir(self, root_dir: Path, lang_code: str) -> None:
        """Load .cha files by Language."""
        all_items = []
        for item in root_dir.rglob("*.cha"):
            file_id = str(item.relative_to(root_dir).parent).replace("/", "_")
            file_id = f"{file_id}_{item.stem}"
            all_items.append((file_id, item))
        self.dataset[lang_code] = all_items

    def export(self, location: Path, *, show_progress: bool = False) -> None:
        """Export dataset to directory."""
        for lang, filelist in self.dataset.items():
            lang_loc = location / lang
            lang_loc.mkdir(parents=True, exist_ok=True)
            metadata = []

            for name, file in track(
                filelist, description=f"Processing {lang} .cha files...", disable=not show_progress
            ):
                try:
                    data = parsing.cha.extract(file, name)  # type: ignore[call-arg]
                    (lang_loc / "child").mkdir(parents=True, exist_ok=True)
                    (lang_loc / "adult").mkdir(parents=True, exist_ok=True)

                    (lang_loc / "child" / f"{name}.raw").write_text("\n".join(data.child_speech))
                    (lang_loc / "adult" / f"{name}.raw").write_text("\n".join(data.adult_speech))
                    metadata.append(data.csv_entry())
                except UnicodeDecodeError:
                    print(f"Failed to process {file}...", file=sys.stderr)
                    raise
            df = pd.DataFrame(metadata, columns=["file_id", "lang", "child_gender", "child_age"])
            df.to_csv(lang_loc / "metadata.csv", index=False)

    def export_turn_taking(self, location: Path) -> None:
        """Export dataset in the turntaking format to given directory."""
        for lang, filelist in self.dataset.items():
            lang_loc = location / lang
            lang_loc.mkdir(parents=True, exist_ok=True)
            for name, file in filelist:
                try:
                    data = parsing.cha.extract_with_tags(file, name)  # type: ignore[call-arg]
                    (lang_loc / "txt").mkdir(parents=True, exist_ok=True)

                    as_json = json.dumps(data.speech, indent=4, default=utils.default_json_encoder)
                    (lang_loc / "txt" / f"{name}.json").write_text(as_json)

                except UnicodeDecodeError:
                    print(f"Failed to process {file}...", file=sys.stderr)
                    raise


class OrganizeByAge:
    """Organize child-speech by age."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def make_age_splits(self, lang_code: str) -> pd.DataFrame:
        """Create a split of metadata.csv into age groups."""
        data_dir = self.root_dir / lang_code
        metadata_df = pd.read_csv(data_dir / "metadata.csv", sep=",")
        metadata_df["child_age(float)"] = metadata_df["child_age"].apply(parse_childes_age)

        # Group by age groups
        metadata_df["age_group"] = pd.cut(
            metadata_df["child_age(float)"],
            bins=range(settings.CHILDES.MAX_AGE),
            labels=[f"{m}_{n}" for m, n in settings.CHILDES.AGE_RANGES],
        )

        # Remove ages outside of range
        return metadata_df[~metadata_df["age_group"].isna()]

    def build_splits(self, lang_code: str, metadata: pd.DataFrame) -> None:
        """Build folders separating children by age."""
        data_dir = self.root_dir / lang_code
        target = data_dir / "child_by_age"
        target.mkdir(exist_ok=True, parents=True)

        for min_age, max_age in settings.CHILDES.AGE_RANGES:
            curr_range = f"{min_age}_{max_age}"
            files_list = list(metadata[metadata["age_group"] == curr_range]["file_id"])
            # Make directory
            (target / curr_range).mkdir(exist_ok=True, parents=True)

            # Create symlink for all files
            for file_id in files_list:
                (target / curr_range / f"{file_id}.txt").symlink_to(data_dir / "child" / f"{file_id}.txt")
