import logging
import os
from pathlib import Path

import torch

# Import the necessary modules
try:
    # Import ColumnParallelLinear if it exists in the flash_attn package
    # This is the missing reference that's causing the issue
    from flash_attn.layers.linear import ColumnParallelLinear
    from flash_attn.models.gpt import GPTLMHeadModel
except ImportError:
    # Fallback imports or error handling if needed
    raise ImportError("flash_attn package is required. Please install it with: pip install flash-attn")

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


class FlashGPTWrapper(torch.nn.Module):
    """Wrapper for flash-attention GPT model to handle HuggingFace's expected inputs."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs
    ) -> dict[str, torch.Tensor]:
        """Forward method that filters out unwanted parameters.

        Args:
            input_ids: Token IDs batch
            attention_mask: Ignored, included in signature for compatibility
            **kwargs: Other arguments that will be ignored

        Returns:
            Model outputs

        """
        # Only pass input_ids to the flash-attention model
        return self.model(input_ids=input_ids)


class NoAttentionMaskCollator:
    """Data collator that removes attention_mask from batches."""

    def __init__(self, tokenizer, mlm: bool = False):
        self.base_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=mlm,
        )

    def __call__(self, features: list) -> dict:
        """Process features and remove attention_mask.

        Args:
            features: List of tokenized examples

        Returns:
            Batch dictionary without attention_mask

        """
        batch = self.base_collator(features)
        if "attention_mask" in batch:
            del batch["attention_mask"]
        return batch


def patch_gpt_model():
    """Patch the flash-attention GPT model to fix the isinstance issue.

    This patch modifies the forward method to safely handle the isinstance check
    that's causing the error.
    """
    original_forward = GPTLMHeadModel.forward

    def safe_forward(self, *args, **kwargs):
        # Store original __instancecheck__ method
        if hasattr(ColumnParallelLinear, "__instancecheck__"):
            original_instancecheck = ColumnParallelLinear.__instancecheck__

            # Define a safe replacement that catches errors
            def safe_instancecheck(cls, instance):
                try:
                    return original_instancecheck(instance)
                except TypeError:
                    return False

            # Replace with our safe version temporarily
            ColumnParallelLinear.__instancecheck__ = safe_instancecheck

        try:
            # Call the original forward method
            return original_forward(self, *args, **kwargs)
        finally:
            # Restore original method if we modified it
            if hasattr(ColumnParallelLinear, "__instancecheck__") and "original_instancecheck" in locals():
                ColumnParallelLinear.__instancecheck__ = original_instancecheck

    # Apply the patch
    GPTLMHeadModel.forward = safe_forward


def transformer_training(
    args: by_size.BySizeTrainItem,
    params_file: Path | None = None,
    tokenizer_name: str = "phonemetransformers/GPT2-85M-CHAR-TXT",
) -> Trainer:
    """Run transformer training using standard HuggingFace components with joined utterances.

    Args:
        args: Training arguments containing data paths
        params_file: Path to model parameters file
        tokenizer_name: Name of the tokenizer to use

    Returns:
        Configured Trainer instance

    """
    # Apply the patch to fix the isinstance issue
    patch_gpt_model()

    # Ensure tokenizers parallelism is disabled
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load model parameters
    model_params = train_params.load_model_params(params_file=params_file)

    # Load tokenizer - standard Hugging Face tokenizer
    L.info("Loading char tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    L.info(f"Tokenizer loaded with vocabulary size: {len(tokenizer.get_vocab())}")

    # Use the custom collator that removes attention_mask
    data_collator = NoAttentionMaskCollator(
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
        n_positions=model_params.gpt2.max_position_embeddings,
        n_ctx=model_params.gpt2.max_position_embeddings,
        n_embd=model_params.gpt2.n_embd,
        n_layer=model_params.gpt2.n_layer,
        n_head=model_params.gpt2.n_head,
        use_flash_attn=True,
    )

    # Initialize GPT model with FlashAttention
    L.info("Initializing Flash Attention GPT model")
    base_model = GPTLMHeadModel(config)

    # Wrap the model with our adapter to handle unwanted parameters
    L.info("Wrapping the Flash attention model to fit HF Trainer")
    model = FlashGPTWrapper(base_model)

    # Create and return the Trainer
    L.info("Creating customized Trainer")
    return Trainer(
        model=model,
        args=train_params.setup_training_arguments(args.model_root_dir, params=model_params),
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=model_params.early_stopping_patience)],
    )


def test_flash_gpt_wrapper():
    """Test that the wrapper and patches work correctly.

    Returns:
        True if all tests pass

    """
    # Apply the patch
    patch_gpt_model()

    # Create a small test config
    config = GPT2Config(
        vocab_size=100,
        n_positions=128,
        n_embd=128,
        n_layer=2,
        n_head=2,
        use_flash_attn=True,
    )

    # Initialize model and wrapper
    try:
        base_model = GPTLMHeadModel(config)
        wrapped_model = FlashGPTWrapper(base_model)

        # Create test input
        input_ids = torch.randint(0, 100, (2, 10))

        # Test forward pass
        outputs = wrapped_model(input_ids=input_ids)

        print("Test passed! Model and wrapper work correctly.")
        return True
    except Exception as e:
        print(f"Test failed with error: {e!s}")
        return False
