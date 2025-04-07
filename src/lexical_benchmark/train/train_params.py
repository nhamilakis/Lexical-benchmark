import logging
import typing as t
from pathlib import Path

import IPython
from pydantic import BaseModel
from transformers import TrainingArguments

L = logging.getLogger(__name__)


class GPT2Params(BaseModel):
    """Parameters specific to GPT2 Trainer."""

    block_size: int
    max_seq_length: int
    mlm: bool
    early_stopping_patience: int

    vocab_size: int
    max_position_embeddings: int
    n_head: int
    n_layer: int
    n_embd: int
    n_inner: int


class LSTMParams(BaseModel):
    """Parameters specific to LSTM Trainer."""

    block_size: int
    max_seq_length: int
    mlm: bool
    early_stopping_patience: int
    hidden_size: int
    vocab_size: int
    embedding_dim: int
    num_layers: int
    dropout: float


class ModelParams(BaseModel):
    """Parameters for model training."""

    overwrite_output_dir: bool
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    warmup_steps: int
    warmup_ratio: float
    lr_scheduler_type: str
    optim: str
    adam_beta1: float
    adam_beta2: float
    weight_decay: float
    max_grad_norm: float
    logging_steps: float
    save_strategy: str
    save_steps: int
    save_total_limit: int
    evaluation_strategy: str
    eval_steps: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    fp16: bool
    dataloader_num_workers: int
    disable_tqdm: bool
    early_stopping_patience:int
    lstm: LSTMParams
    gpt2: GPT2Params

    def for_training_args(self) -> dict[str, t.Any]:
        """Export arguments for training."""
        return self.model_dump(exclude={"lstm", "gpt2"})


def load_model_params(params_file: Path | None = None) -> ModelParams:
    """Load model params either from input or from default."""
    params_file = params_file if params_file else Path(__file__).parent / "train-params.toml"

    L.info(f"Loading params from {params_file}")
    data = params_file.read_toml()
    return ModelParams(**data)


def setup_training_arguments(model_path: Path, params: ModelParams) -> TrainingArguments:
    """Configure training arguments to match Fairseq settings."""
    return TrainingArguments(output_dir=str(model_path), **(params.for_training_args()))


if __name__ == "__main__":
    parameters = load_model_params()
    IPython.embed()
