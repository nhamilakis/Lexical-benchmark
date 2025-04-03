import logging
import typing as t
from pathlib import Path

import torch
from transformers import (
    EarlyStoppingCallback,
    GPT2LMHeadModel,
    PretrainedConfig,
    Trainer,
)

from lexical_benchmark.dataloaders import by_size
from lexical_benchmark.train import tokenizers, train_params

if t.TYPE_CHECKING:
    from lexical_benchmark.train.trainers import TrainerP


L = logging.getLogger(__name__)


class CustomGPT2Config(PretrainedConfig):
    """Configuration class for gpt2 language model."""

    model_type = "gpt2"

    def __init__(
        self,
        gpt_params: train_params.GPT2Params,
        **kwargs,
    ) -> None:
        """Initialize gpt2 Config."""
        super().__init__(**kwargs)
        self.vocab_size = kwargs.get("vocab_size", gpt_params.vocab_size)
        self.max_position_embeddings = gpt_params.max_position_embeddings
        self.n_head = gpt_params.n_head  # Number of attention heads
        self.n_layer = gpt_params.n_layer  # Number of hidden layers
        self.n_embd = gpt_params.n_embd  # Hidden size (embedding dimension)
        self.n_inner = gpt_params.n_inner  # Dimension of the feedforward network


def transformer_training(args: by_size.BySizeTrainItem, params_file: Path | None = None) -> "TrainerP":
    """Run transformer training on current arguments."""
    model_params = train_params.load_model_params(params_file=params_file)
    # Load tokenizer and create data collator
    L.info("Loading char-tokenizer")
    tokenizer = tokenizers.load_char_tokenizer()

    L.info("Character tokenizer has been loaded")
    data_collator = tokenizers.CustomDataCollatorForLanguageModeling(
        tokenizer, max_seq_length=model_params.gpt2.max_seq_length, mlm=model_params.gpt2.mlm
    )
    L.info(f"Vocabulary size: {len(tokenizer.get_vocab())}")

    L.info("Tokenizing the dataset")
    dataset = tokenizers.load_dataset(args.train_txt, args.dev_txt)

    data_preprocessor = tokenizers.DataPreprocessor(tokenizer)

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

    L.info("Loading configurations & initialising GPT2 model trainer")
    config = CustomGPT2Config(gpt_params=model_params.gpt2, vocab_size=len(tokenizer.get_vocab()))

    model = GPT2LMHeadModel(config)
    return Trainer(
        model=model,
        args=train_params.setup_training_arguments(args.model_root_dir, params=model_params),
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=model_params.lstm.early_stopping_patience)],
    )
