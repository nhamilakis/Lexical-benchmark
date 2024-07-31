from pathlib import Path

import pandas as pd

from lexical_benchmark.datasets import utils


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

    def export_to_dir(self, location: Path) -> None:
        """Export dataset to directory."""
        for lang, filelist in self.dataset.items():
            lang_loc = location / lang
            lang_loc.mkdir(parents=True, exist_ok=True)
            metadata = []

            for name, file in filelist:
                data = utils.extract_from_cha(file, name)
                (lang_loc / "child" ).mkdir(parents=True, exist_ok=True)
                (lang_loc / "adult" ).mkdir(parents=True, exist_ok=True)

                (lang_loc / "child" / f"{name}.txt").write_text("\n".join(data.child_speech))
                (lang_loc / "adult" / f"{name}.txt").write_text("\n".join(data.adult_speech))
                metadata.append(data.csv_entry())
            df = pd.DataFrame(metadata, columns=["file_id", "lang", "child_gender", "child_age"])
            df.to_csv(lang_loc / "metadata.csv", index=False)


