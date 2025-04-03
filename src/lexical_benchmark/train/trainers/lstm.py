import logging
import typing as t
from pathlib import Path

import torch
from torch import nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)

from lexical_benchmark.dataloaders import by_size
from lexical_benchmark.train import tokenizers, train_params

if t.TYPE_CHECKING:
    from lexical_benchmark.train.trainers import TrainerP


L = logging.getLogger(__name__)


class LSTMConfig(PretrainedConfig):
    """Configuration class for LSTM language model."""

    model_type = "lstm"

    def __init__(
        self,
        lstm_params: train_params.LSTMParams,
        **kwargs,
    ) -> None:
        """Initialize LSTM Config."""
        super().__init__(**kwargs)
        self.vocab_size = lstm_params.vocab_size
        self.embedding_dim = lstm_params.embedding_dim
        self.hidden_size = lstm_params.hidden_size
        self.num_layers = lstm_params.num_layers
        self.dropout = lstm_params.dropout


class LSTMForLanguageModeling(PreTrainedModel):
    """LSTM-based language model compatible with HuggingFace's interface."""

    config_class = LSTMConfig

    def __init__(self, config: LSTMConfig) -> None:
        super().__init__(config)

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
        )
        self.output = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,  # noqa: ARG002
        labels: torch.LongTensor | None = None,
        *,
        return_dict: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward."""
        embeddings = self.embedding(input_ids)
        lstm_output, _ = self.lstm(embeddings)
        logits = self.output(lstm_output)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
            }
        return (loss, logits)


def lstm_training(args: by_size.BySizeTrainItem, params_file: Path | None = None) -> "TrainerP":
    """Run LSTM training on current arguments."""
    from transformers import (
        EarlyStoppingCallback,
        Trainer,
    )

    model_params = train_params.load_model_params(params_file=params_file)

    # Load tokenizer and create data collator
    # TODO: check added tokens on bySizeTrain
    L.info("Loading char-tokenizer")
    tokenizer = tokenizers.load_char_tokenizer()

    L.info("Character tokenizer has been loaded")
    data_collator = tokenizers.CustomDataCollatorForLanguageModeling(
        tokenizer, max_seq_length=model_params.lstm.max_seq_length, mlm=model_params.lstm.mlm
    )
    L.info(f"Vocabulary size: {len(tokenizer.get_vocab())}")

    L.info("Tokenizing the dataset")
    dataset = tokenizers.load_dataset(args.train_txt(), args.dev_txt())
    data_preprocessor = tokenizers.DataPreprocessor(tokenizer)
    # TODO: this uses tqdm (disable it)
    processed_dataset = dataset.map(
        data_preprocessor,
        batched=True,
        num_proc=(64 if torch.cuda.is_available() else 1),
        remove_columns=["text"],
    )
    train_dataset = processed_dataset["train"]
    val_dataset = processed_dataset["valid"]

    L.info(f"Training dataset size: {len(train_dataset)}")
    L.info(f"Validation dataset size: {len(val_dataset)}")

    L.info("Loading configurations & initialising LSTM model trainer")
    model = LSTMForLanguageModeling(
        config=LSTMConfig(lstm_params=model_params.lstm, vocab_size=len(tokenizer.get_vocab()))
    )
    return Trainer(
        model=model,
        args=train_params.setup_training_arguments(args.model_root_dir, params=model_params),
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=model_params.lstm.early_stopping_patience)],
    )
    # TODO: we should write configs somewhere in the target dir ?
