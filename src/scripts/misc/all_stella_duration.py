#!/home/nhamilakis/envs/venvs/lbenchmark/bin/python3.11
# fmt: off
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=stela-audio-duration
#SBATCH --time=2:00:00
#SBATCH --export=ALL
#SBATCH --output stela-duration-%J.log
# fmt: on

import typing as t
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from lexical_benchmark.datasets import stella
from lexical_benchmark.utils import slurm_utils, timed_status

slurm_utils.info_header()


def get_wav_durations(files_iter: t.Iterable[Path]) -> dict[str, float]:
    """Get durations of all WAV files in directory."""
    durations = {}
    for wav_file in files_iter:
        duration = librosa.get_duration(path=str(wav_file))
        durations[wav_file.name] = duration
    return durations


def audio_mapping(wav_loc: Path, file_type: str = ".wav") -> dict[str, Path]:
    """Get map to audio names to path."""
    return {f"{wav.name}": wav for wav in wav_loc.rglob(f"*{file_type}")}


def crawl_dataset() -> pd.DataFrame:
    """Crawl all STELA dataset & calculate audio durations."""
    stela_dt = stella.STELATranscriptDataset()
    data_prep = stella.InfTrainStructure(
        root_dir=stela_dt.source_path,
        metadata_dir=stela_dt.meta.meta_root_path,
        lang="EN",
    )
    audio_path = audio_mapping(wav_loc=stela_dt.source_path / "wav")
    durations_list = []
    for audio in data_prep.iter_wavs():
        file = audio_path.get(audio.wav)
        duration = librosa.get_duration(path=str(file)) if file else np.nan

        durations_list.append(
            (audio.language, audio.hour, audio.split, audio.speaker, audio.book, audio.wav, duration),
        )
    cols = ["lang", "hour", "split", "section", "book", "wav", "duration"]
    return pd.DataFrame(durations_list, columns=cols)


def extract_book_times(duration_df: pd.DataFrame) -> pd.DataFrame:
    """Group by book-times."""
    book_summary = duration_df[duration_df["hour"] == "3200h"]
    return (
        book_summary.groupby("book")
        .agg(
            {
                "lang": "first",
                "wav": "count",
                "duration": "sum",
            }
        )
        .rename(columns={"wav": "file_count"})
    )


def main() -> None:
    """Main function."""
    print("Crawling though stela & computing durations...")
    with timed_status(status="Crawing STELA/EN", complete_status="Completed Computing Durations.."):
        df_duration = crawl_dataset()
    df_duration.to_csv("data-v2/datasets/STELATranscriptions/metadata/stela_audio_duration.csv", index=False, sep=";")

    book_times = extract_book_times(df_duration)
    book_times.to_csv("data-v2/datasets/STELATranscriptions/metadata/book_audio_durations.csv", sep=";")


if __name__ == "__main__":
    main()

# Job Done
slurm_utils.info_footer()
