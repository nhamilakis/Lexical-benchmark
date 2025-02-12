#!/usr/bin/env python
import json
import typing as t

from rich.console import Console

from lexical_benchmark.datasets import wordstats

console = Console()

def contains_uppercase(text: str) -> bool:
    """Check if string contains at least one uppercase character."""
    return any(char.isupper() for char in text)


def without_keys(data: dict[str, t.Any], keys_to_remove: list[str]) -> dict[str, t.Any]:
    """Remove multiple keys from a dictionary."""
    return {k: v for k, v in data.items() if k not in keys_to_remove}


def fix_upper(pos_map: dict[str, list[str]]) -> dict[str, list[str]]:
    """Fix POS map dictionairy words that contain upper cases.

    Any word that contains an upper case needs to be lowercased and
    reinserted into the dictionairy.

    If the word pre-exists we need to merge the two entries after lowercasing
    the first to avoid overwriting information.
    """
    # Extract all words that have upper-case characters
    upper_keys = [k for k in pos_map if contains_uppercase(k)]
    corrected_keys = {}  # Words corrected to be lowercase
    for key in upper_keys:
        tag_count: list[str] = pos_map[key]
        lower_key = key.lower()
        # Append pre-existing lower item if it is present
        tag_count.extend(pos_map.get(lower_key, []))
        # add in the corrected keys
        corrected_keys[lower_key] = tag_count

    # Make a dict of pos_map without all the upper
    pos_map_clean = without_keys(data, upper_keys)
    # Add the corrected values
    pos_map_clean.update(corrected_keys)

    return pos_map_clean


if __name__ == "__main__":
    wd_dataset = wordstats.WordStatsDataset()

    console.print("Loading STELA...")
    with wd_dataset.pos_maps.stela.open() as fd:
        data = json.load(fd)

    with console.status("Fixing STELA"):
        clean_data = fix_upper(data)

    console.print("writing STELA ")
    with wd_dataset.pos_maps.stela.open("w") as fd:
        json.dump(clean_data, fd, indent=4)

    # ChildRealistic
    console.print("Loading ChildRealistic...")
    with wd_dataset.pos_maps.child_realistic.open() as fd:
        data = json.load(fd)

    with console.status("Fixing ChildRealistic"):
        clean_data = fix_upper(data)

    console.print("writing ChildRealistic ")
    with wd_dataset.pos_maps.child_realistic.open("w") as fd:
        json.dump(clean_data, fd, indent=4)

    print("Completed")
