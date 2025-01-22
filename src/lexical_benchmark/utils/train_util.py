#!/usr/bin/env python
import logging
import os
from pathlib import Path
import string

import torch
from lexical_benchmark.utils import hf_util
from torch import nn
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    LineByLineTextDataset, 
    PreTrainedTokenizer
)



def tokenize_data(tokenizer, data_path, block_size: int):
    """Tokenize the dataset."""
    return LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=data_path,
        block_size=block_size,
    )



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


