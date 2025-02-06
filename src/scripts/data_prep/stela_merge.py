#!/home/nhamilakis/envs/venvs/lbenchmark/bin/python3.11
# fmt: off
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=stela-cleanups
#SBATCH --time=5:00:00
#SBATCH --export=ALL
#SBATCH --output stela-clean-%J.log
# fmt: on
"""Build script for STELA/txt_merged.

This script helps build the `txt_merged` folder in the STELA dataset, which creates the same
chunking as the original but its build from the ground up by merging two 50h chunks to create a 100h one, etc..

That way we know that the totals are always the same.
"""
from lexical_benchmark import settings
from lexical_benchmark.datasets import stella

######
# ARGS
######
lang = "EN"
root_dir = settings.PATH.dataset_root / "STELATranscriptions2"

#####
dataset = stella.STELATranscriptDataset(root_dir=root_dir)
location = dataset.root_dir / "txt_merged" / lang

# 50h
print("Building 50h...")
loc = location / "50h"
for s in dataset.sections(lang=lang, hour_split="50h"):
    text = dataset.item(lang, "50h", s).clean.transcription.read_text()
    (loc / s / "transcription.txt").safe_write_text(text)

# 100h
print("Building 100h...")
sections = list(dataset.sections(lang=lang, hour_split="50h"))
loc = location / "100h"
step = 2
for count, i in enumerate(range(0, len(sections), step)):
    group = sections[i : i + step]
    text = ""
    for idx in group:
        item = dataset.item(lang, "50h", idx)
        text += " " + item.clean.transcription.read_text()
    (loc / f"{count:02d}" / "transcription.txt").safe_write_text(text)


# 200h
print("Building 200h...")
sections = list(dataset.sections(lang=lang, hour_split="50h"))
loc = location / "200h"
count = 0
step = 4
for count, i in enumerate(range(0, len(sections), step)):
    group = sections[i : i + step]
    text = ""
    for idx in group:
        item = dataset.item(lang, "50h", idx)
        text += " " + item.clean.transcription.read_text()
    (loc / f"{count:02d}" / "transcription.txt").safe_write_text(text)


# 400h
print("Building 400h...")
sections = list(dataset.sections(lang=lang, hour_split="50h"))
loc = location / "400h"
count = 0
step = 8
for count, i in enumerate(range(0, len(sections), step)):
    group = sections[i : i + step]
    text = ""
    for idx in group:
        item = dataset.item(lang, "50h", idx)
        text += " " + item.clean.transcription.read_text()
    (loc / f"{count:02d}" / "transcription.txt").safe_write_text(text)


# 800h
print("Building 800h...")
sections = list(dataset.sections(lang=lang, hour_split="50h"))
loc = location / "800h"
count = 0
step = 16
for count, i in enumerate(range(0, len(sections), step)):
    group = sections[i : i + step]
    text = ""
    for idx in group:
        item = dataset.item(lang, "50h", idx)
        text += " " + item.clean.transcription.read_text()
    (loc / f"{count:02d}" / "transcription.txt").safe_write_text(text)


# 1600h
print("Building 1600h...")
sections = list(dataset.sections(lang=lang, hour_split="50h"))
loc = location / "1600h"
count = 0
step = 32
for count, i in enumerate(range(0, len(sections), step)):
    group = sections[i : i + step]
    text = ""
    for idx in group:
        item = dataset.item(lang, "50h", idx)
        text += " " + item.clean.transcription.read_text()
    (loc / f"{count:02d}" / "transcription.txt").safe_write_text(text)


# 3200h
print("Building 3200h...")
sections = list(dataset.sections(lang=lang, hour_split="50h"))
loc = location / "1600h"
count = 0
step = 64
for count, i in enumerate(range(0, len(sections), step)):
    group = sections[i : i + step]
    text = ""
    for idx in group:
        item = dataset.item(lang, "50h", idx)
        text += " " + item.clean.transcription.read_text()
    (loc / f"{count:02d}" / "transcription.txt").safe_write_text(text)


print(f"Completed Merge Build for {root_dir} !")
