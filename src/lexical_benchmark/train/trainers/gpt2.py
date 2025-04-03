import logging
import typing as t
from pathlib import Path

from transformers import (
    EarlyStoppingCallback,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
)

from lexical_benchmark.dataloaders import by_size
from lexical_benchmark.train import tokenizers, train_params

if t.TYPE_CHECKING:
    from lexical_benchmark.train.trainers import TrainerP


L = logging.getLogger(__name__)


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
    dataset = tokenizers.load_dataset(args.train_txt(), args.dev_txt())

    data_preprocessor = tokenizers.DataPreprocessor(tokenizer)

    processed_dataset = dataset.map(
        data_preprocessor,
        batched=True,
        num_proc=8,
        remove_columns=["text"],
    )
    L.info("map finished ?")
    train_dataset = processed_dataset["train"]
    val_dataset = processed_dataset["valid"]

    L.info(f"Training dataset size: {len(train_dataset)}")
    L.info(f"Validation dataset size: {len(val_dataset)}")

    L.info("Loading configurations & initialising GPT2 model trainer")
    config = GPT2Config(
        vocab_size=len(tokenizer.get_vocab()),
        max_position_embeddings=model_params.gpt2.max_position_embeddings,
        n_head=model_params.gpt2.n_head,
        n_layer=model_params.gpt2.n_layer,
        n_embd=model_params.gpt2.n_embd,
        n_inner=model_params.gpt2.n_inner,
    )
    model = GPT2LMHeadModel(config)
    return Trainer(
        model=model,
        args=train_params.setup_training_arguments(args.model_root_dir, params=model_params),
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=model_params.gpt2.early_stopping_patience)],
    )
