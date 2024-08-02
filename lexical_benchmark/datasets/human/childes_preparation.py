import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.progress import track

from lexical_benchmark import settings
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
                *.txt
            adult_speech/
                *.txt
        EN_NA/
            metadata.csv
            child_speech/
                *.txt
            adult_speech/
                *.txt
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

    def export_to_dir(self, location: Path, *, show_progress: bool = False) -> None:
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

                    (lang_loc / "child" / f"{name}.txt").write_text("\n".join(data.child_speech))
                    (lang_loc / "adult" / f"{name}.txt").write_text("\n".join(data.adult_speech))
                    metadata.append(data.csv_entry())
                except UnicodeDecodeError:
                    print(f"Failed to process {file}...", file=sys.stderr)
                    raise
            df = pd.DataFrame(metadata, columns=["file_id", "lang", "child_gender", "child_age"])
            df.to_csv(lang_loc / "metadata.csv", index=False)


class OrganizeByAge:
    """Organize child-speech by age."""

    def __init__(self, root_dir: Path, lang_codes: list[str]) -> None:
        self.root_dir = root_dir
        self.lang_codes = lang_codes

    def make_age_splits(self) -> None:
        """Create a split of metadata.csv into age groups."""
        for lang in self.lang_codes:
            data_dir = self.root_dir / lang
            metadata_df = pd.read_csv(data_dir / "metadata.csv", sep=",")
            metadata_df["child_age(float)"] = metadata_df["child_age"].apply(parse_childes_age)

            for min_age, max_age in settings.CHILDES.AGE_RANGES:
                metadata_df[
                    (metadata_df["child_age(float)"] >= min_age) & (metadata_df["child_age(float)"] < max_age)
                ].to_csv(data_dir / f"child_{min_age}_{max_age}.csv", index=False, sep=",")

    def build_splits(self) -> None:
        """Build folders separating children by age."""
        for lang in self.lang_codes:
            data_dir = self.root_dir / lang
            target = data_dir / "child_by_age"

            # iterate over csv containing splits
            for file in data_dir.glob("child_*.csv"):
                df = pd.read_csv(file, sep=",")
                age_range = target / file.stem.replace("child_", "")
                age_range.mkdir(exist_ok=True, parents=True)
                for row in df.itertuples():
                    (age_range / f"{row.file_id}.txt").symlink_to(data_dir / "child" / f"{row.file_id}.txt")
