#!/home/nhamilakis/envs/venvs/lbenchmark/bin/python3.11
# fmt: off
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --job-name=asr-sampler
#SBATCH --time=1:00:00
#SBATCH --export=ALL
#SBATCH --output testasr-%J.log
# fmt: on


import argparse
import random
from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf
import whisper

from lexical_benchmark.utils import slurm_utils

slurm_utils.info_header()


class ASRSampler:
    """Class performing random sampling on a list of audios and extractring transcriptions."""

    def __init__(self, whisper_model: str = "base", output_dir: Path = Path("segments")) -> None:
        """Initialize processor with whisper model and output directory.

        Args:
            whisper_model: Whisper model size ("tiny", "base", "small", "medium", "large")
            output_dir: Directory to save segments and transcriptions

        """
        self.model = whisper.load_model(whisper_model)
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def extract_random_segments(
        self,
        audio_files: list[Path],
        segment_duration: float,  # In seconds
        num_segments: int,
        sr: int = 16000,  # Sample Rate
    ) -> list[tuple[npt.NDArray[np.float32], int]]:
        """Extract random segments from audio files."""
        segment_length = int(segment_duration * sr)
        segments: list[tuple[npt.NDArray[np.float32], int]] = []

        while len(segments) < num_segments:
            file_path = random.choice(audio_files)
            audio, _ = librosa.load(file_path, sr=sr)

            if len(audio) < segment_length and len(audio) < 5:
                segment = audio
            else:
                max_start = len(audio) - segment_length
                if max_start <= 0:
                    continue
                start_idx = random.randint(0, max_start)
                segment = audio[start_idx : start_idx + segment_length]

            segments.append((segment, sr))

        return segments

    def transcribe_segments(self, segments: list[tuple[npt.NDArray[np.float32], int]]) -> list[dict[str, str]]:
        """Transcribe audio segments using Whisper."""
        transcriptions = []
        for segment, _ in segments:
            result = self.model.transcribe(segment)
            transcriptions.append(result)
        return transcriptions

    def save_transcribed_segments(
        self, segments: list[tuple[npt.NDArray[np.float32], int]], transcriptions: list[dict[str, str]]
    ) -> None:
        """Save audio segments and transcriptions to files."""
        for idx, ((segment, sr), trans) in enumerate(zip(segments, transcriptions, strict=False)):
            segment_path = self.output_dir / f"seg{idx}.wav"
            trans_path = self.output_dir / f"seg{idx}.txt"

            sf.write(str(segment_path), segment, sr)
            trans_path.write_text(trans["text"])


if __name__ == "__main__":
    #  "/scratch1/projects/lexical-benchmark/v2/datasets/STELATranscriptions/src/original/wav/EN/3595/5788_LibriVox_en"
    parser = argparse.ArgumentParser()
    parser.add_argument("location", type=str, help="Audio location")
    parser.add_argument("-t", "--file-extension", type=str, default=".wav", help="Audio type to search for")
    parser.add_argument("-n", "--number-of-segments", default=5, type=int, help="Number of segments to create.")
    parser.add_argument("-d", "--segment-duration", type=float, default=25.0, help="Max duration of each segment.")
    parser.add_argument("-s", "--sample-rate", type=int, default=16000, help="Sample rate of audio files.")
    parser.add_argument("-o", "--output-dir", type=str, default="data-v2/asr-test", help="Dir to save results.")
    parser.add_argument("-m", "--whisper-model", type=str, default="base", help="Whisper model to use.")
    args = parser.parse_args()
    processor = ASRSampler(whisper_model=args.whisper_model, output_dir=Path(args.output_dir))
    audio_dir = Path(args.location)
    audio_files = list(audio_dir.rglob(f"*{args.file_extension}"))
    print(f"{len(audio_files)=}")
    segments = processor.extract_random_segments(
        audio_files,
        segment_duration=args.segment_duration,
        num_segments=args.number_of_segments,
        sr=args.sample_rate,
    )
    print(f"Extracted {len(segments)} segments from {len(audio_files)}")
    transcriptions = processor.transcribe_segments(segments)
    print(f"Extracted {len(transcriptions)} transcriptions")
    processor.save_transcribed_segments(segments, transcriptions)
    print(f"Transcriptions saved to {processor.output_dir}...")


# Job Done
slurm_utils.info_footer()
