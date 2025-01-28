from pathlib import Path
import argparse

def parseargs():
    # Run parameters
    parser = argparse.ArgumentParser(description="rename folders")
    parser.add_argument(
        "--source_dir", 
        type=str, 
        default="/scratch1/projects/lexical-benchmark/v2/gen/merged", 
        help="relative path to the root dir"
    )
    parser.add_argument(
        "--source_filename", 
        type=str, 
        default="gen.csv", 
        help="source filename to be modified"
    )
    parser.add_argument(
        "--target_filename", 
        type=str, 
        default="1000_hour_per_year.csv", 
        help="source filename to be modified"
    )
    return parser.parse_args()

def rename_files(base_path: Path | str, source_filename: str, target_filename: str) -> dict[Path, Path]:
    """Find and rename all source files to target files."""
    try:
        # Find all gen.csv files recursively
        for file_path in base_path.rglob(source_filename):
            new_path = file_path.parent / target_filename
            # If target already exists, create a unique name
            file_path.rename(new_path)
            print(f"replacing the source name {file_path} to {new_path}")
    except Exception as e:
        print(f"Error processing files: {e}")

def main() -> None:
    """Test the file renaming functionality."""
    # Args parser
    args = parseargs()
    # Perform renaming
    rename_files(Path(args.source_dir), args.source_filename, args.target_filename)

if __name__ == "__main__":
    main()