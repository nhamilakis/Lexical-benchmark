import logging
import typing as t

import wandb

from lexical_benchmark import train_lib
from lexical_benchmark.utils import generic as generic_utils

# Forward reference for optuna.Trial since we don't import optuna
OptunaTrial = t.TypeVar("OptunaTrial", bound="optuna.Trial")


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


def lstm_training(args: train_lib.TrainArgs, logger: logging.Logger) -> Trainer:
    """Run LSTM training on current arguments."""
    from transformers import (
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
        Trainer,
    )

    # TODO: migrate magick numbers to a model config loader
    block_size = 128
    model_max_length = 2048
    logger.info("Loading char-tokenizer")

    # Load tokenizer and create data collator
    tokenizer = train_lib.hf_tools.load_char_tokenizer(
        model_max_length=model_max_length, special_token_lst=args.AddedTokens
    )
    print("Character tokenizer has been loaded")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    logger.info(f"Vocabulary size: {len(tokenizer.get_vocab())}")

    logger.info("Tokenizing the dataset")
    train_dataset = train_lib.tokenize_data(tokenizer, args.train_text_path, block_size)
    val_dataset = train_lib.tokenize_data(tokenizer, args.dev_text_path, block_size)
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    logger.info("Loading configurations & initialising LSTM model trainer")
    config = train_lib.LSTMConfig(vocab_size=len(tokenizer.get_vocab()))
    # TODO: standardize model config loader
    model = train_lib.LSTMForLanguageModeling(config)
    return Trainer(
        model=model,
        args=train_lib.setup_training_arguments(args.current_model_path),
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )


def transformer_training(args: train_lib.TrainArgs, logger: logging.Logger) -> Trainer:
    """Run transformer training on current arguments."""
    from transformers import (
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
        GPT2Config,
        GPT2LMHeadModel,
        Trainer,
    )

    # TODO: migrate magick numbers to a model config loader
    block_size = 128
    model_max_length = 2048

    # Configure the decoder-only transformer model
    config = GPT2Config(
        vocab_size=58,
        max_position_embeddings=1024,
        n_head=8,  # Number of attention heads
        n_layer=3,  # Number of hidden layers
        n_embd=1024,  # Hidden size (embedding dimension)
        n_inner=4096,  # Dimension of the feedforward network
    )  # Hidden size (embedding dimension)

    logger.info("Loading char-tokenizer")
    # Load tokenizer and create data collator
    tokenizer = train_lib.load_char_tokenizer(model_max_length=model_max_length, special_token_lst=args.AddedTokens)
    print("Character tokenizer has been loaded")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    logger.info(f"Vocabulary size: {len(tokenizer.get_vocab())}")

    print("Tokenizing the dataset")
    train_dataset = train_lib.tokenize_data(tokenizer, args.train_text_path, block_size)
    val_dataset = train_lib.tokenize_data(tokenizer, args.dev_text_path, block_size)
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    logger.info("Loading configurations & initialising GPT2 model trainer")
    # Initialize the model with the configured settings
    model = GPT2LMHeadModel(config=config)

    # Initialize trainer
    return Trainer(
        model=model,
        args=train_lib.setup_training_arguments(args.current_model_path),
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )


def main(args: train_lib.TrainArgs) -> None:
    """Main function acting as a CMD."""
    # Create output directory if it doesn't exist
    args.current_model_path.mkdir(exist_ok=True, parents=True)

    ####
    # Setup logging
    if args.log_to_std:
        generic_utils.setup_logging(args.log_level)
    else:
        generic_utils.setup_logging(args.log_level, log_file=args.log_path, no_stdout=True)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with arguments: {args.to_dict()}")

    ####
    # Setup wandb
    wandb.init(
        project="Lexical-Benchmark",
        # name format: datasetname_model_month_chunk  e.g. child_lstm_2_00
        name=args.job_name,
        mode="offline",
    )
    logger.info(f"Wandb job name: {args.job_name}")

    logger.info(f"Loading trainer for {args.model_type}")
    if args.model_type == "lstm":
        trainer = lstm_training(args, logger)
    elif args.model_type == "transformer":
        trainer = transformer_training(args, logger)
    else:
        raise ValueError(f"Undefined model_type = {args.model_type}")

    resume_file = args.get_resume_path()
    # Run Training
    if resume_file:
        logger.info(f"Resuming previous training using {resume_file}")

    logger.info(f"Running training of {args.job_name}")
    trainer.train(resume_from_checkpoint=str(resume_file) if resume_file else None)

    # Save the final model
    logger.info(f"Training of {args.job_name} has completed !")
    trainer.save_model(str(args.current_model_path))
    logger.info(f"Model saved to {args.current_model_path}")
