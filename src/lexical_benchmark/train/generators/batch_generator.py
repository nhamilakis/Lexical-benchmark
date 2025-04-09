import typing as t
from pathlib import Path

from lexical_benchmark import lb_types

Model = t.Any


class BatchGenerator:
    """Generator class."""

    def __init__(self, *, model_path: Path, use_vllm: bool, model_type: lb_types.MODEL_TYPE) -> None:
        self.model = self.load_mode(model_path=model_path, use_vllm=use_vllm, model_type=model_type)

    def load_model(self, *, model_path: Path, use_vllm: bool, model_type: lb_types.MODEL_TYPE) -> Model:  # noqa: ARG002
        """Load the model."""
        if model_type == "lstm":
            # TODO: load model
            return ...
        if model_type == "gpt2":
            # TODO: load model
            return ...
        raise ValueError(f"Unknown model type {model_type}")

    def generate_items(self, nb_tokens: int, target_file: Path, *, resume: bool = True, override: bool = False) -> None:
        """Generate data from model."""
        # TODO: implement generation
