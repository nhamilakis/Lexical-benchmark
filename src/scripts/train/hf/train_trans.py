#!/usr/bin/env python  # noqa: EXE001
import argparse
import contextlib
import logging
from pathlib import Path

import wandb
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
)

from lexical_benchmark import settings
from lexical_benchmark.utils.hf_util import load_char_tokenizer
from lexical_benchmark.utils.train_util import setup_training_arguments, tokenize_data


def parseargs() -> argparse.Namespace:
    """Argument parser from CMD."""
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")
    parser.add_argument("dataset", choices=["ChildRealistic", "STELATranscriptions2"])
    parser.add_argument("split", type=int, help="The name of the split (ex: 1, 2, etc...)")
    parser.add_argument("chunk", type=int, help="The number of the chunk (ex: 0, 1, etc..)")
    parser.add_argument("validation_path", help="Path to the validation file")


    # Optional Arguments
    parser.add_argument("--data-root", default=settings.PATH.DATA_DIR)  # ROOT of where the data is
    parser.add_argument("--output-name", default="models")
    parser.add_argument("--input-name", default="datasets")
    parser.add_argument("--data-type", choices=["by_month", "txt"], default="by_month")
    parser.add_argument("--lang", default="EN")

    parser.add_argument("--resume", action="store_true", help="Whether to resume from previous ckpt: True or False")
    parser.add_argument("--AddedTokens", default=["'", "|"], help="A list of added special tokens")
    return parser.parse_args()


# largest size of each block
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


def build_path(args: argparse.Namespace, folder: str) -> Path:
    """Helps build a path to a given location."""
    return Path(args.data_root) / folder / args.dataset \
            / args.data_type / args.lang / f"{args.split:02}" / f"{args.chunk:02}"


def main() -> None:
    # Args parser
    args = parseargs()
    # Path locations
    model_path = build_path(args, args.output_name) / "trans"
    train_path = build_path(args, args.input_name) / "char_hf.txt"
    validation_path = Path(args.validation_path)


    # Create output directory if it doesn't exist
    model_path.mkdir(exist_ok=True, parents=True)

    print("##################")
    print("Setting up logging")
    print("##################")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(model_path /"training.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting training with arguments: %s", args)

    # init weight and biases
    job_name = f"{args.dataset}_tran_{args.split:02}_{args.chunk:02}"
    wandb.init(
    project="Lex_benchmark",
    # name format: datasetname_model_month_chunk  e.g. child_lstm_2_00
    name=job_name,
    mode="offline"
    )
    print(f'Wandb job name: {job_name}')

    print("######################")
    print("Loading char-tokenizer")
    print("######################")

    # Load tokenizer and create data collator
    tokenizer = load_char_tokenizer(model_max_length=2048, special_token_lst=args.AddedTokens)
    print("Character tokenizer has been loaded")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    logger.info(f"Vocabulary size: {len(tokenizer.get_vocab())}")

    print("######################")
    print("Tokenizing the dataset")
    print("######################")

    train_dataset = tokenize_data(tokenizer, train_path, block_size)
    val_dataset = tokenize_data(tokenizer, validation_path, block_size)
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    print("#################")
    print("Loading the model")
    print("#################")

    # Initialize the model with the configured settings
    model = GPT2LMHeadModel(config=config)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=setup_training_arguments(model_path),
        data_collator=data_collator,
        train_dataset=train_dataset,  # You'll need to implement dataset loading
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("##############")
    print("Start training")
    print("##############")

    if args.resume:
        # Resume training if checkpoint specified
        ckpt_lst = []
        for ckpt in model_path.iterdir():
            if ckpt.is_dir():
                with contextlib.suppress(Exception):
                    ckpt_lst.append(int(ckpt.name.split("-")[1]))
        try:
            resume_path = f"{model_path}/checkpoint-{str(max(ckpt_lst))}"
            trainer.train(resume_from_checkpoint=resume_path)
            print(f"Resuming ckpt from {resume_path}")
        except:
            print("No checkpoint to resume. Train model from scratch!")
            trainer.train()
    else:
        trainer.train()
        print("Training the Transformer model from scratch!")

    # Save the final model
    trainer.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
