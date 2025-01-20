from typing import Optional, Dict, Any
from dataclasses import dataclass
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from torch import nn
import torch
import os
import logging
import argparse
from pathlib import Path


from transformers import GPT2Config, GPT2LMHeadModel
import os
import sys
import string
import argparse
from lexical_benchmark.utils.hf_util import *
from lexical_benchmark.utils.format_util import str_to_bool
import wandb

wandb.init(mode="offline")


def parseargs():
    # Run parameters
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")
    parser.add_argument(
        "--TrainPath",
        type=str,
        default="/scratch1/projects/lexical-benchmark/v2/datasets/ChildRealistic/by_month/EN/36/00/char_hf.txt",
        help="Path to the train file",
    )
    parser.add_argument(
        "--ValPath",
        type=str,
        default="/scratch1/projects/lexical-benchmark/v2/datasets/ChildRealistic/dev/EN/char_hf.txt",
        help="Path to the validation file",
    )
    parser.add_argument(
        "--OutPath",
        type=str,
        default="/scratch1/projects/lexical-benchmark/v2/models/ChildRealistic/by_month/EN/36/00",
        help="Directory to save model checkpoints",
    )
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


def setup_training_arguments(args) -> TrainingArguments:
    """Configure training arguments to match Fairseq settings."""
    return TrainingArguments(
        output_dir=args.OutPath,
        overwrite_output_dir=True,
        # Batch size and optimization
        per_device_train_batch_size=32,  # Increase batch size
        gradient_accumulation_steps=4,  # Increase gradient accumulation steps
        max_steps=100000,  # Reduce max steps
        # Learning rate schedule
        learning_rate=1e-4,
        warmup_steps=1000,
        warmup_ratio=0.0,
        lr_scheduler_type="inverse_sqrt",
        # Optimizer settings
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.98,
        weight_decay=0.01,
        max_grad_norm=0.0,
        # Logging and saving
        logging_dir=args.OutPath,
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=20,  # Reduce save total limit
        # Evaluation
        evaluation_strategy="steps",
        eval_steps=1000,
        # Early stopping settings
        load_best_model_at_end=True,  # Required for early stopping
        metric_for_best_model="eval_loss",  # Monitor eval loss for early stopping
        greater_is_better=False,  # Lower loss is better
        # FP16 training
        fp16=True,  # Match Fairseq's fp16
        # Misc
        dataloader_num_workers=4,
        disable_tqdm=False,
    )


def main(argv):
    # Args parser
    args = parseargs()
    # Create output directory if it doesn't exist
    Path(args.OutPath).mkdir(exist_ok=True, parents=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(os.path.join(args.OutPath, "training.log")), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting training with arguments: %s", args)

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

    train_dataset = tokenize_data(tokenizer, args.TrainPath, block_size)
    val_dataset = tokenize_data(tokenizer, args.ValPath, block_size)
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
        args=setup_training_arguments(args),
        data_collator=data_collator,
        train_dataset=train_dataset,  # You'll need to implement dataset loading
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer = Trainer(
        model=model,
        args=setup_training_arguments(args),
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Assuming you have a validation set
    )

    print("##############")
    print("Start training")
    print("##############")

    if args.resume:
        # Resume training if checkpoint specified
        ckpt_lst = []
        for ckpt in Path(args.OutPath).iterdir():
            if ckpt.is_dir():
                try:
                    ckpt_lst.append(int(ckpt.name.split("-")[1]))
                except:
                    pass
        try:
            resume_path = f"{args.OutPath}/checkpoint-{str(max(ckpt_lst))}"
            trainer.train(resume_from_checkpoint=resume_path)
            print(f"Resuming ckpt from {resume_path}")
        except:
            print("No checkpoint to resume. Train model from scratch!")
            trainer.train()
    else:
        trainer.train()
        print("Training the Transformer model from scratch!")

    # Save the final model
    trainer.save_model(args.OutPath)
    logger.info(f"Model saved to {args.OutPath}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
