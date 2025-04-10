import logging
import os
from pathlib import Path

from flash_attn.models.gpt import GPTLMHeadModel
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    GPT2Config,
    Trainer,
)

from lexical_benchmark.dataloaders import by_size
from lexical_benchmark.train import tokenizers, train_params

# Set up logging
L = logging.getLogger(__name__)


def transformer_training(
    args: by_size.BySizeTrainItem,
    params_file: Path | None = None,
    tokenizer_name: str = "phonemetransformers/GPT2-85M-CHAR-TXT",
) -> Trainer:
    """Run transformer training using standard HuggingFace components with joined utterances."""
    # Ensure tokenizers parallelism is disabled
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load model parameters
    model_params = train_params.load_model_params(params_file=params_file)

    # Load tokenizer - standard Hugging Face tokenizer
    L.info("Loading char tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    L.info(f"Tokenizer loaded with vocabulary size: {len(tokenizer.get_vocab())}")

    # Use standard data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Use standard autoregressive LM (not masked LM)
    )

    # Load datasets with joined utterances to maximize context usage
    L.info(f"Loading and processing datasets with joined utterances (max_length={model_params.gpt2.max_seq_length})")
    train_dataset = tokenizers.load_joined_text(args.train_txt(), tokenizer, model_params.gpt2.max_seq_length)
    val_dataset = tokenizers.load_joined_text(args.dev_txt(), tokenizer, model_params.gpt2.max_seq_length)

    L.info(f"Training dataset size: {len(train_dataset)}")
    L.info(f"Validation dataset size: {len(val_dataset)}")

    # Load GPT2 configuration
    L.info("Creating GPT2 configuration")
    config = GPT2Config(
        vocab_size=len(tokenizer.get_vocab()),
        n_positions=model_params.gpt2.max_position_embeddings,  # Use standard parameter name
        n_ctx=model_params.gpt2.max_position_embeddings,  # Context size matches position embeddings
        n_embd=model_params.gpt2.n_embd,  # Embedding dimension from params
        n_layer=model_params.gpt2.n_layer,  # Number of layers from params
        n_head=model_params.gpt2.n_head,  # Number of attention heads from params
    )

    # intialize GPT model with FlashAttention
    model = GPTLMHeadModel(config)
    # Create Trainer
    L.info("Creating standard Trainer")
    return Trainer(
        model=model,
        args=train_params.setup_training_arguments(args.model_root_dir, params=model_params),
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=model_params.early_stopping_patience)],
    )
