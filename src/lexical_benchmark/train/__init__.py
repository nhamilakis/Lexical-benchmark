from clypi import Command, Positional
from transformers import AutoTokenizer


class _DownloadHFCMD(Command):
    """Download a model from HF."""

    model_name: Positional[str]

    def download_hf_pretrained(self) -> None:
        """Download a model using hugging-face API."""


def hf_dl_cmd() -> None:
    """Command-Line Entrypoint for downloader."""
    # Download Auto-Encoder
    AutoTokenizer.from_pretrained("phonemetransformers/GPT2-85M-CHAR-TXT")
