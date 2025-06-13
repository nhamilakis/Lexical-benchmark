import logging
import typing as t
from pathlib import Path

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
)

from lexical_benchmark import lb_types
from lexical_benchmark.dataloaders import by_size
from lexical_benchmark.dataloaders import childes as childes_loaders
from lexical_benchmark.train import tokenizers, train_params

if t.TYPE_CHECKING:
    from lexical_benchmark.train.trainers import TrainerP

import os

T = t.TypeVar("T")
TrainerP = t.TypeVar("TrainerP", bound=Trainer)


L = logging.getLogger(__name__)


class LSTMConfig(PretrainedConfig):
    """Configuration class for LSTM language model."""

    model_type = "lstm"

    def __init__(
        self,
        vocab_size=None,
        embedding_dim=None,
        hidden_size=None,
        num_layers=None,
        dropout=None,
        lstm_params=None,
        **kwargs,
    ) -> None:
        """Initialize LSTM Config."""
        super().__init__(**kwargs)

        # If lstm_params is provided, use those values
        if lstm_params is not None:
            self.vocab_size = lstm_params.vocab_size
            self.embedding_dim = lstm_params.embedding_dim
            self.hidden_size = lstm_params.hidden_size
            self.num_layers = lstm_params.num_layers
            self.dropout = lstm_params.dropout
        else:
            # Otherwise use the provided individual parameters or defaults
            self.vocab_size = vocab_size or 10000
            self.embedding_dim = embedding_dim or 300
            self.hidden_size = hidden_size or 512
            self.num_layers = num_layers or 2
            self.dropout = dropout or 0.1


class LSTMForLanguageModeling(PreTrainedModel):
    """LSTM-based language model compatible with HuggingFace's interface."""

    config_class = LSTMConfig

    def __init__(self, config: LSTMConfig, device: lb_types.DEVICE_TYPE = "cuda") -> None:
        """Initialize the LSTM model for language modeling."""
        super().__init__(config)

        self.to(device=device)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim).to(device=device)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
        ).to(device)
        self.output = nn.Linear(config.hidden_size, config.vocab_size).to(device)

        # Move the entire model to the device after initialization
        self.to(device)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,  # noqa: ARG002
        labels: torch.LongTensor | None = None,
        *,
        return_dict: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the LSTM model."""
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

    def _forward_single_step(
        self, input_ids: torch.LongTensor, hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for single token with hidden state management."""
        embeddings = self.embedding(input_ids)
        if hidden_state is None:
            lstm_output, new_hidden = self.lstm(embeddings)
        else:
            lstm_output, new_hidden = self.lstm(embeddings, hidden_state)
        logits = self.output(lstm_output)
        return logits, new_hidden

    def _generate_with_sampling(
        self, input_ids: torch.LongTensor, max_length: int, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0
    ) -> torch.LongTensor:
        """Generate using stateful LSTM as contexts have been inherently incorporated in the states."""
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Initialize hidden state
        h0 = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_size, device=device)
        c0 = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_size, device=device)
        hidden_state = (h0, c0)

        generated = input_ids.clone()

        with torch.no_grad():
            # Process initial sequence to build hidden state
            if input_ids.shape[1] > 0:
                _, hidden_state = self._forward_single_step(input_ids, hidden_state)

            # Generate new tokens one by one (statefully)
            for _ in range(max_length - input_ids.shape[1]):
                # Only process the last token with maintained hidden state
                last_token = generated[:, -1:] if generated.shape[1] > 0 else generated
                logits, hidden_state = self._forward_single_step(last_token, hidden_state)

                next_token_logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

        return generated

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int,
        *,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        **kwargs,  # noqa: ARG002
    ) -> torch.LongTensor:
        """Generate text tokens using the LSTM model."""
        if do_sample:
            return self._generate_with_sampling(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        return None


def lstm_training(
    args: by_size.BySizeTrainItem | childes_loaders.CHILDESTrainItem,
    params_file: Path | None = None,
    tokenizer_name: str = "phonemetransformers/GPT2-85M-CHAR-TXT",
    batch_size: int = 128,
    device: lb_types.DEVICE_TYPE = "cuda",
) -> Trainer:
    """Run transformer training using standard HuggingFace components with joined utterances."""
    # Ensure tokenizers parallelism is disabled
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load model parameters
    model_params = train_params.load_model_params(params_file=params_file)
    model_params.per_device_train_batch_size = batch_size

    # Load tokenizer - standard Hugging Face tokenizer
    L.info("Loading char-tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    L.info(f"Tokenizer loaded with vocabulary size: {len(tokenizer.get_vocab())}")

    # Use standard data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Use standard autoregressive LM (not masked LM)
    )

    # Load datasets with joined utterances to maximize context usage
    L.info(f"Loading and processing datasets with joined utterances (max_length={model_params.lstm.max_seq_length})")
    train_dataset = tokenizers.load_joined_text(args.train_txt(), tokenizer, model_params.lstm.max_seq_length)
    val_dataset = tokenizers.load_joined_text(args.dev_txt(), tokenizer, model_params.lstm.max_seq_length)

    L.info(f"Training dataset size: {len(train_dataset)}")
    L.info(f"Validation dataset size: {len(val_dataset)}")

    # Create LSTM model
    L.info("Loading configurations & initialising LSTM model trainer")
    model = LSTMForLanguageModeling(
        config=LSTMConfig(lstm_params=model_params.lstm, vocab_size=len(tokenizer.get_vocab())), device=device
    )

    return Trainer(
        model=model,
        args=train_params.setup_training_arguments(args.model_root_dir, params=model_params),
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=model_params.early_stopping_patience)],
    )
