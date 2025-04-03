import typing as t
from pathlib import Path

from lexical_benchmark import exc
from lexical_benchmark.dataloaders import by_size

# Forward reference for optuna.Trial since we don't import optuna
OptunaTrial = t.TypeVar("OptunaTrial", bound="optuna.Trial")  # type: ignore[undefined-variable]# noqa: F821


class TrainerP(t.Protocol):
    """Protocol defining the interface for trainers."""

    def train(
        self,
        resume_from_checkpoint: str | None,
        trial: OptunaTrial | dict[str, t.Any] | None = None,
        ignore_keys_for_eval: list[str] | None = None,
        **kwargs: t.Any,
    ) -> None:
        """Train the model.

        Raises:
            ValueError: If resume_from_checkpoint path doesn't exist
            RuntimeError: If training fails

        """
        ...

    def save_model(self, output_dir: str | None = None) -> None:
        """Saves the final model."""
        ...


def load_trainer(item: by_size.BySizeTrainItem, *, model_params_file: Path | None = None) -> TrainerP:
    """Load model trainer."""
    match item.model_type:
        case "lstm":
            from .lstm import lstm_training

            return lstm_training(args=item, params_file=model_params_file)
        case "gpt2":
            from .gpt2 import transformer_training

            return transformer_training(args=item, params_file=model_params_file)
        case _:
            raise exc.BadModelTypeError(f"Model {item.model_type} not in given model list !")
