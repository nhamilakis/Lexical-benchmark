#!/usr/bin/env python
"""merge gen from JZ and obeorn"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from lexical_benchmark import settings


def parseargs():
    # Run parameters
    parser = argparse.ArgumentParser(description="Get the array script for generation")
    parser.add_argument("--source1", type=str, default="gen/oberon", help="Source Directory1")
    parser.add_argument("--source2", type=str, default="gen/jz", help="Directory to save path file")
    parser.add_argument("--dest", type=str, default="gen/merged", help="Destination Directory to save path file")
    parser.add_argument("--strategy", type=str, default="size", help="Merging strategy")
    return parser.parse_args()


def merge_folders(src_folder1, src_folder2, dst_folder, strategy="overwrite"):
    """Merge folders with conflict resolution.

    Args:
        src_folder1 (str): Path to first source folder
        src_folder2 (str): Path to second source folder
        dst_folder (str): Path to destination folder
        strategy (str): Conflict resolution strategy ('ask', 'newest', 'skip', 'overwrite')
    """
    Path(dst_folder).mkdir(parents=True, exist_ok=True)

    def get_file_info(path):
        return {"mtime": datetime.fromtimestamp(path.stat().st_mtime), "size": path.stat().st_size}

    def resolve_conflict(file1, file2, rel_path):
        if strategy == "newest":
            # get the newest file
            info1 = get_file_info(file1)
            info2 = get_file_info(file2)
            return file2 if info2["mtime"] > info1["mtime"] else file1
        elif strategy == "largest":
            # get the largest file
            info1 = get_file_info(file1)
            info2 = get_file_info(file2)
            return file2 if info2["size"] > info1["size"] else file1
        elif strategy == "skip":
            return None
        elif strategy == "overwrite":
            return file2
        elif strategy == "ask":
            while True:
                choice = input(
                    f"\nConflict for {rel_path}\n"
                    f"1: Keep version from folder1 ({file1})\n"
                    f"2: Keep version from folder2 ({file2})\n"
                    f"s: Skip this file\n"
                    "Choose [1/2/s]: "
                )
                if choice == "1":
                    return file1
                if choice == "2":
                    return file2
                if choice.lower() == "s":
                    return None
        return None

    # Process both folders
    for item in Path(src_folder1).glob("**/*"):
        if item.is_file():
            rel_path = item.relative_to(src_folder1)
            dst_path = Path(dst_folder) / rel_path
            src2_path = Path(src_folder2) / rel_path

            dst_path.parent.mkdir(parents=True, exist_ok=True)

            if src2_path.exists():
                # Conflict found
                resolved_path = resolve_conflict(item, src2_path, rel_path)
                if resolved_path:
                    shutil.copy2(resolved_path, dst_path)
                    print(f"Resolved conflict for: {rel_path}")
            else:
                # No conflict
                shutil.copy2(item, dst_path)
                print(f"Copied from folder1: {rel_path}")

    # Copy remaining files from folder2
    for item in Path(src_folder2).glob("**/*"):
        if item.is_file():
            rel_path = item.relative_to(src_folder2)
            dst_path = Path(dst_folder) / rel_path
            if not dst_path.exists():
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst_path)
                print(f"Copied from folder2: {rel_path}")


def main():
    # Args parser
    args = parseargs()
    root_dir: Path = settings.PATH.DATA_DIR

    folder1 = f"{root_dir}/{args.source1}"
    folder2 = f"{root_dir}/{args.source2}"
    destination = f"{root_dir}/{args.dest}"

    # Choose strategy: 'ask', 'newest', 'skip', or 'overwrite'
    merge_folders(folder1, folder2, destination, strategy="overwrite")


if __name__ == "__main__":
    main()
