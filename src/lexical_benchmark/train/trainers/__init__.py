import typing as t

from lexical_benchmark import exc
from lexical_benchmark.lb_types import MODEL_TYPE

# Forward reference for optuna.Trial since we don't import optuna
OptunaTrial = t.TypeVar("OptunaTrial", bound="optuna.Trial")  # type: ignore[undefined-variable]# noqa: F821


class Trainer(t.Protocol):
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


def load_trainer(model_type: MODEL_TYPE) -> Trainer:
    """Load model trainer."""
    match model_type:
        case "lstm":
            from .lstm import TrainClass

            return ...
        case "gpt2":
            from .gpt2 import TrainClass

            return TrainClass
        case _:
            raise exc.BadModelTypeError(f"Model {model_type} not in given model list !")
