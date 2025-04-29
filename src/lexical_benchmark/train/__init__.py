from clypi import Command, Positional


class _DownloadHFCMD(Command):
    """Download a model from HF."""

    model_name: Positional[str]

    def download_hf_pretrained(self) -> None:
        """Download a model using hugging-face API."""


def hf_dl_cmd() -> None:
    """Command-Line Entrypoint for HF pre-downloader."""
    from transformers import AutoTokenizer

    # Download Auto-Encoder
    AutoTokenizer.from_pretrained("phonemetransformers/GPT2-85M-CHAR-TXT")
