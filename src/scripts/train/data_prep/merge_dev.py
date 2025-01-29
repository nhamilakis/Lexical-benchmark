"""Merge dev set"""

from pathlib import Path

from lexical_benchmark import settings

file_mode = "test"
lang = "EN"
root_dir: Path = settings.PATH.dataset_root / "STELATranscriptions2"
source_dir = root_dir / "src" / "preprocessed" / "txt" / lang
data_location = root_dir / file_mode / lang
filename = "transcription.txt"

data_all = []
for file in source_dir.iterdir():
    if file.name.endswith(f"{file_mode}.preprocessed"):
        with open(file, "r") as f:
            print(file)
            data = f.readlines()
            data_all.extend(data)

print("Finished concatenating all the files")


# write out the results
with open(data_location / filename, "w") as f:
    for text in data_all:
        f.write(text)

print(f"Finished writing file to {data_location / filename}")
