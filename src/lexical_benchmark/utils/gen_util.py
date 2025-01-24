import logging
import random
import string
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

from lexical_benchmark.utils import hf_util


class Logger:
    """Utility class for logging configuration."""

    @staticmethod
    def setup(output_dir: str, filename: str = "inference.log") -> logging.Logger:
        """Configure and return a logger."""
        log_path = Path(output_dir) / filename
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.FileHandler(str(log_path)), logging.StreamHandler()],
            force=True,
        )
        return logging.getLogger(__name__)


class TextGenerator:
    """Handles text generation using transformer models."""

    def __init__(
        self,
        model_path: str,
        model_max_length: int = 1024,
        model_type: str = "transformer",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the generator.
        model_path: Path to the model
        chars: String containing all characters for tokenization
        model_max_length: Maximum sequence length
        model_type: Type of model ("transformer" or "lstm")
        """
        # Initialize tokenizer first
        chars = string.ascii_letters
        self.tokenizer = hf_util.CharacterTokenizer(chars=chars, model_max_length=model_max_length)
        self.model = self._load_model(model_path, model_type, device)
        self.device = device
        self.model.eval()
        self.max_length = min(model_max_length, getattr(self.model.config, "n_positions", model_max_length))

    def _load_model(self, model_path: str, model_type: str, device: str) -> PreTrainedModel:
        """Load the model from path."""
        try:
            if model_type.lower() == "lstm":
                config = LSTMConfig.from_pretrained(model_path)
                model = LSTMForLanguageModeling.from_pretrained(model_path, config=config)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path)
            print("Model has been loaded")
            return model.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    # TODO: merge it with hf_util
    def add_special_tokens(self, special_token_lst: list[str] = ["'", "|"]):
        # add the new token for abbreviation
        for special_token in special_token_lst:
            self.tokenizer.add_tokens(special_token)
        print(f"Added {len(special_token_lst)} special tokens to the tokenizer")

    def generate_text(self, word_num: int, temp_lst: list[float]) -> dict[str, str]:
        """Generate text with different temperatures."""
        results = {}
        # Use a valid token ID from the tokenizer's vocabulary
        random_token_id = random.randint(0, 25)

        for temp in temp_lst:
            with torch.no_grad():
                input_ids = torch.tensor([[random_token_id]], device=self.device)
                gen = self.tokenizer.decode([random_token_id])
                bar_count = 0

                while bar_count < word_num and input_ids.shape[1] < self.max_length:
                    curr_length = input_ids.shape[1]
                    position_ids = torch.arange(curr_length, device=self.device).unsqueeze(0)

                    outputs = self.model.generate(
                        input_ids=input_ids,
                        max_length=curr_length + 1,  # make sure generating one token each time
                        do_sample=True,
                        temperature=temp,
                        num_beams=1,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        position_ids=position_ids,
                        use_cache=True,
                    )

                    new_token = outputs[0, -1].item()
                    decoded_token = self.tokenizer.decode([new_token])

                    if decoded_token == "|":
                        bar_count += 1

                    gen += decoded_token
                    input_ids = outputs

                    if input_ids.shape[1] >= self.max_length - 2:
                        input_ids = input_ids[:, :1]

                results[f"unprompted_{temp}"] = gen

        return results


class BatchProcessor:
    """Handles batch processing of text generation."""

    def __init__(self, generator: TextGenerator, save_path: Path, chunk_size: int = 10) -> None:
        """Initialize batch processor."""
        self.generator = generator
        self.save_path = Path(save_path)
        self.chunk_size = chunk_size
        self.logger = Logger.setup(self.save_path)

    def process_batch(self, batch: pd.DataFrame, temp_lst: list[float]) -> pd.DataFrame:
        """Process a single batch of data."""
        temp_columns = [f"unprompted_{temp}" for temp in temp_lst]
        results = []

        for _, row in batch.iterrows():
            try:
                result = self.generator.generate_text(row["sent_len"], temp_lst)
                results.append(pd.Series(result))
            except Exception as e:
                self.logger.exception(f"Error in generation: {str(e)}")
                results.append(pd.Series({col: "" for col in temp_columns}))
            finally:
                torch.cuda.empty_cache()

        batch[temp_columns] = pd.DataFrame(results, index=batch.index)
        return batch

    def segment_df(self,source_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
        """Select rows to be generated and match source/ref dataframes."""
        # Get columns after 'model' column
        gen_cols = source_df.columns.tolist()[source_df.columns.get_loc("model") + 1 :]
        generated_df = pd.DataFrame()
        gen_mat_df = pd.DataFrame()
        # Group by sentence length
        for sent_len, ref_df_group in ref_df.groupby("sent_len"):
            # Filter source rows matching current length
            source_gen = source_df[source_df["sent_len"] == sent_len]
            row_num = min(source_gen.shape[0], ref_df_group.shape[0])
            # Combine matched rows from ref and source
            generated_df = pd.concat(
                [generated_df, pd.concat([ref_df_group.head(row_num), source_gen[gen_cols].head(row_num)], axis=1)]
            )
            # Store unmatched ref rows for generation
            if source_gen.shape[0] < ref_df_group.shape[0]:
                gen_mat_df = pd.concat([gen_mat_df, ref_df_group.tail(len(ref_df_group) - len(source_gen))])
        return generated_df, gen_mat_df

    def process_dataframe(self, df: pd.DataFrame, temp_lst: list[float], resume: bool = False) -> pd.DataFrame:
        """Process entire dataframe with save intermediate generations and updated databases."""
        gen = pd.DataFrame()
        resume_file = self.save_path / "gen_intermediate.csv"

        if resume and resume_file.is_file():
            source_df = pd.read_csv(resume_file).loc[:, "month":]
            # select the rows to be generated
            gen, df = self.segment_df(source_df, df)
            self.logger.info(
                f"Resuming from previous generation. Rows processed: {len(gen)} \
                    Generating {len(df)} of sentences"
            )

        total_rows = len(df)
        chunks = [df.iloc[i : i + self.chunk_size] for i in range(0, total_rows, self.chunk_size)]

        # generate the rest of the marterials
        for chunk in tqdm(chunks, desc="Processing chunks"):
            processed_chunks = []

            for i in range(0, len(chunk), self.chunk_size):
                batch = chunk.iloc[i : i + self.chunk_size].copy()
                processed_batch = self.process_batch(batch, temp_lst)
                processed_chunks.append(processed_batch)
                torch.cuda.empty_cache()

            processed_df = pd.concat(processed_chunks)
            gen = pd.concat([gen, processed_df])
            if resume and resume_file.is_file():
                source_df = pd.concat([source_df, processed_df])
                # Save intermediate results
                source_df.to_csv(self.save_path / "gen_intermediate.csv")
            self.logger.info(f"Saved intermediate results. Total rows processed: {len(gen)}")
        # return the generated dataset matched in quantity and sent length
        return gen


class LSTMConfig(PretrainedConfig):
    """Configuration class for LSTM language model."""

    model_type = "LSTM"

    def __init__(
        self,
        vocab_size: int = 58,
        embedding_dim: int = 200,
        hidden_size: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
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
        self.embedding = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = torch.nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
        )
        self.output = torch.nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=True):
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
        num_return_sequences: int = 1,
        **kwargs,
    ) -> torch.LongTensor:
        """Generate text tokens using the LSTM model."""
        if do_sample:
            return self._generate_with_sampling(input_ids, max_length, temperature, top_k, top_p, num_return_sequences)
        else:
            return self._generate_greedy(input_ids, max_length, num_return_sequences)

    def _generate_with_sampling(self, input_ids, max_length, temperature, top_k, top_p, num_return_sequences):
        """Temperature sampling for LM generations."""
        generated = input_ids.clone()

        for _ in range(max_length - input_ids.shape[1]):
            outputs = self(input_ids=generated)
            next_token_logits = outputs["logits"][:, -1, :] / temperature

            if top_k > 0:
                # Apply top-k filtering
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")

            if top_p < 1.0:
                # Apply nucleus (top-p) sampling
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
