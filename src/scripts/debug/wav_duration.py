#!/home/nhamilakis/envs/venvs/lbenchmark/bin/python3.11
# fmt: off
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=audio-duration
#SBATCH --time=1:00:00
#SBATCH --export=ALL
#SBATCH --output duration-%J.log
# fmt: on

import csv
import typing as t
from pathlib import Path

import librosa
import numpy as np
from tap import Tap

from lexical_benchmark.utils import slurm_utils

slurm_utils.info_header()


def get_wav_durations(files_iter: t.Iterable[Path]) -> dict[str, float]:
    """Get durations of all WAV files in directory."""
    durations = {}
    for wav_file in files_iter:
        duration = librosa.get_duration(path=str(wav_file))
        durations[wav_file.name] = duration
    return durations


def save_durations(durations: dict[str, float], output_file: Path) -> None:
    """Save durations to CSV file."""
    with output_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "duration_seconds"])
        for filename, duration in sorted(durations.items()):
            writer.writerow([filename, f"{duration:.2f}"])


class WavDuration(Tap):
    """Command line for argument parser."""

    directory: Path  # Search directory
    output_file: Path | None = None  # ouptut files
    recursive: bool = True  # Recursively search directory
    sum: bool = True  # Print sum duration
    file_type: str = ".wav"  # File types to search


def main() -> None:
    """Main function."""
    args = WavDuration().parse_args()
    if args.recursive:
        files_iter = Path(args.directory).rglob(f"*{args.file_type}")
    else:
        files_iter = Path(args.directory).glob(f"*{args.file_type}")

    durations = get_wav_durations(files_iter)

    if args.output_file is not None:
        output_file = Path(args.output_file)
        save_durations(durations, output_file)

    if args.sum:
        total = np.sum(list(durations.values()))
        print(f"Total duration for {args.directory}:\nFound {len(durations)} audio files with a duration of {total}s")


if __name__ == "__main__":
    main()

# Job Done
slurm_utils.info_footer()
