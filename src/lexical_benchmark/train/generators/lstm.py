import logging
import typing as t

import torch
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
)

if t.TYPE_CHECKING:
    from lexical_benchmark.train.trainers import TrainerP


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

    def __init__(self, config: LSTMConfig):
        super().__init__(config)
        self.embedding = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = torch.nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
        )
        self.output = torch.nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, return_dict=True):
        embeddings = self.embedding(input_ids)
        lstm_output, _ = self.lstm(embeddings)
        logits = self.output(lstm_output)

        if return_dict:
            return {"logits": logits}
        return (logits,)

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        **kwargs,
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

    def _generate_with_sampling(
        self, input_ids: torch.LongTensor, max_length: int, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0
    ) -> torch.LongTensor:
        """Generate sequences using sampling with temperature, top-k, and top-p filtering."""
        generated = input_ids.clone()
        try:
            with torch.no_grad():
                for _ in range(max_length - input_ids.shape[1]):
                    outputs = self(input_ids=generated)
                    next_token_logits = outputs["logits"][:, -1, :] / temperature

                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float("-inf")

                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        next_token_logits[indices_to_remove] = float("-inf")

                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token], dim=1)

        except Exception as e:
            logging.exception(f"Error in sampling generation: {e!s}")
            raise

        return generated
