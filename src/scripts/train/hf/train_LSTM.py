#!/usr/bin/env python
import argparse
import logging
import os
from pathlib import Path

import torch
import wandb
from lexical_benchmark.utils import hf_util
from lexical_benchmark.utils.format_util import str_to_bool
from torch import nn
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

wandb.init(mode="offline")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LSTM Language Model")
    parser.add_argument(
        "--TrainPath",
        type=str,
        default="/scratch1/projects/lexical-benchmark/v2/datasets/ChildRealistic/by_month/EN/12/00/char_hf.txt",
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
        default="/scratch1/projects/lexical-benchmark/v2/models/ChildRealistic/by_month/EN/12/00",
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--Resume", default="False", help="Whether to resume from previous ckpt: True or False")
    parser.add_argument("--AddedTokens", default=["'", "|"], help="A list of added special tokens")
    return parser.parse_args()


# TODO: add the num_workers and batch_size compatible with new GPU devices

# largest size of each block
block_size = 128
model_max_length = 2048


class LSTMConfig(PretrainedConfig):
    """Configuration class for LSTM language model."""

    model_type = "lstm"

    def __init__(
        self,
        vocab_size: int = 58,
        embedding_dim: int = 200,
        hidden_size: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        """Initialize LSTM Config."""
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout


class LSTMForLanguageModeling(PreTrainedModel):
    """LSTM-based language model compatible with HuggingFace's interface."""

    config_class = LSTMConfig

    def __init__(self, config: LSTMConfig):
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
        attention_mask: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        *,
        return_dict: bool = True,
    ) -> dict[str, torch.Tensor]:
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


def main():
    """Main training function."""
    args = parse_args()

    # Create output directory if it doesn't exist
    Path(args.OutPath).mkdir(exist_ok=True)

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
    tokenizer = hf_util.load_char_tokenizer(model_max_length=2048, special_token_lst=args.AddedTokens)
    print("Character tokenizer has been loaded")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    logger.info(f"Vocabulary size: {len(tokenizer.get_vocab())}")

    print("######################")
    print("Tokenizing the dataset")
    print("######################")

    train_dataset = hf_util.tokenize_data(tokenizer, args.TrainPath, block_size)
    val_dataset = hf_util.tokenize_data(tokenizer, args.ValPath, block_size)
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    print("#################")
    print("Loading the model")
    print("#################")

    # Initialize config and model
    config = LSTMConfig(
        vocab_size=len(tokenizer.get_vocab())  # Should match your vocabulary size
    )
    model = LSTMForLanguageModeling(config)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=setup_training_arguments(args),
        data_collator=data_collator,
        train_dataset=train_dataset,  # You'll need to implement dataset loading
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("##############")
    print("Start training")
    print("##############")

    if str_to_bool(args.Resume):
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
        print("Training the LSTM model from scratch!")

    # Save the final model
    trainer.save_model(args.OutPath)
    logger.info(f"Model saved to {args.OutPath}")


if __name__ == "__main__":
    main()
