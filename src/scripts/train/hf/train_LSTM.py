#!/usr/bin/env python
import argparse
import logging
import os
from pathlib import Path

import wandb
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
)

from lexical_benchmark.settings import dataset_name_dict
from lexical_benchmark.utils import hf_util
from lexical_benchmark.utils.train_util import (
    LSTMConfig,
    LSTMForLanguageModeling,
    setup_training_arguments,
)


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
    parser.add_argument("--resume", action='store_true', help="if true, resume from previous ckpt")
    parser.add_argument("--AddedTokens", default=["'", "|"], help="A list of added special tokens")
    return parser.parse_args()


# largest size of each block
block_size = 128
model_max_length = 2048

#TODO: modify the trainer in train_util to put LSTMConfig here
'''     
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
'''

def main():
    """Main training function."""
    args = parse_args()

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

    model_path = Path(args.OutPath)
    job_name=f"{dataset_name_dict[model_path.parents[3].name]}_lstm_{model_path.parent.name}_{model_path.name}"
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
        vocab_size=len(tokenizer.get_vocab())
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
        print("Training the LSTM model from scratch!")

    # Save the final model
    trainer.save_model(args.OutPath)
    logger.info(f"Model saved to {args.OutPath}")


if __name__ == "__main__":
    main()
