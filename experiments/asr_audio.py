#!/home/nhamilakis/envs/venvs/lbenchmark/bin/python3.11
# fmt: off
#SBATCH --partition=gpu
#SBATCH --nodelist=puck6
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=whisper-transcribe
#SBATCH --time=12:00:00
#SBATCH --export=ALL
#SBATCH --output transcriber-%J.log
# fmt: on


import os
import typing as t
import warnings
from pathlib import Path

import librosa
import whisper
from tap import Tap

from lexical_benchmark.datasets import stella
from lexical_benchmark.utils import slurm_utils

slurm_utils.info_header()

WHISPER_MODEL_TYPE = t.Literal[
    "tiny",
    "tiny.en",  # English Only
    "base",
    "base.en",  # English Only
    "small",
    "small.en",  # English Only
    "medium",
    "medium.en",  # English Only
    "large",
    "turbo",
]


class ASRTranscriber:
    """Class managing ASR Transcriptions of audio files."""

    def __init__(self, model: WHISPER_MODEL_TYPE) -> None:
        self.model = whisper.load_model(model)

    def transcribe(self, audio_file: Path, target: Path, sr: int = 16000) -> None:
        """Transcribe an audio file."""
        audio, _ = librosa.load(audio_file, sr=sr)
        txt = self.model.transcribe(audio)
        try:
            as_segments = [segment["text"] for segment in txt["segments"]]
            target.safe_write_text("\n".join(as_segments))
        except KeyError:
            warnings.warn(f"File {audio_file} failed to transcribe !!", stacklevel=1)


class ASRArgs(Tap):
    """CMD args for ASR."""

    location: str
    file_extension: t.Literal[".wav", ".mp3", ".flac"] = ".wav"  # Extensions to search for
    sample_rate: int = 16000  # Sample Rate of audio
    output_dir: str = "data-v2/asr/"  # Location to write results
    whisper_model: WHISPER_MODEL_TYPE = "small.en"  # Model to use for transcriptions
    save_args: bool = True

    def configure(self) -> None:
        """Extra configuration."""
        self.add_argument("location")


def transcribe_audio_files() -> None:
    """Transcribe given audio files."""
    args_loader = ASRArgs()
    if "ARGS" in os.environ:
        arg_file = Path(os.environ["ARGS"])
        args: ASRArgs = args_loader.from_dict(arg_file.load_json())
    else:
        args = args_loader.parse_args()

    slurm_utils.info_args(args)

    if args.save_args:
        Path("cache").mkdir(exist_ok=True)
        slurm_id = ""
        if "SLURM_JOB_ID" in os.environ:
            slurm_id = "_" + os.environ["SLURM_JOB_ID"]
        args.save(f"cache/args_asr{slurm_id}.json")

    audio_files = Path(args.location).rglob(f"*{args.file_extension}")
    transcriber = ASRTranscriber(model=args.whisper_model)

    for audio in audio_files:
        target = Path(args.output_dir) / f"{audio.stem}.txt"
        transcriber.transcribe(audio, target, sr=args.sample_rate)


def transcribe_stela_books(book_list: list[str]) -> None:
    """Transcribe a stela books."""
    target_dir = Path.cwd() / "data-v2/asr"
    dataset = stella.STELATranscriptDataset()
    audio_map = dataset.book_wav_filemap(lang="EN")
    transcriber = ASRTranscriber(model="small.en")
    prg = Path.cwd() / "transcriber.progress"

    task1 = slurm_utils.ProgressTask(total=len(book_list), task_name="transcribe_books", target_file=prg)
    for book in book_list:
        wav_list = audio_map.get(book, [])
        task_n = slurm_utils.ProgressTask(total=len(wav_list), task_name=f"transcribe_{book}", target_file=prg)
        for audio in wav_list:
            target = target_dir / book / f"{audio.stem}.txt"
            if not target.is_file():
                transcriber.transcribe(audio, target, sr=16000)
            task_n.update()
        task_n.complete()
        task1.update()
    task1.complete()


if __name__ == "__main__":
    book_list = ["4262_LibriVox_en", "6910_LibriVox_en", "4955_LibriVox_en", "5726_LibriVox_en", "5788_LibriVox_en"]
    dataset = stella.STELATranscriptDataset()
    remaining_books = set(dataset.get_books(lang="EN").keys()) - set(book_list)
    transcribe_stela_books(list(remaining_books))
slurm_utils.info_footer()
