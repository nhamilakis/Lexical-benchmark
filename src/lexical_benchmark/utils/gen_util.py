import logging
import random
import string
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from vllm import LLM, SamplingParams

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
        local_rank: int = -1,
        use_vllm: bool = True,
    ):
        # Prevent vLLM for LSTM
        self.use_vllm = use_vllm and model_type.lower() != "lstm"
        self.model_type = model_type
        self.local_rank = local_rank

        # Distributed setup with sync
        if local_rank != -1:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
            self.device = f"cuda:{local_rank}"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Tokenizer setup
        chars = string.ascii_letters
        self.tokenizer = hf_util.CharacterTokenizer(chars=chars, model_max_length=model_max_length)

        # Model initialization with distributed support
        if self.use_vllm:
            self.model = LLM(model=model_path, tokenizer=self.tokenizer)

        else:
            self.model = self._load_model(model_path)
            if local_rank != -1:
                self.model = DistributedDataParallel(self.model, device_ids=[local_rank], find_unused_parameters=True)
            self.model.eval()

        if local_rank != -1:
            dist.barrier()  # Sync after initialization

        self.max_length = min(model_max_length, getattr(self.model.config, "n_positions", model_max_length))

    def _load_model(self, model_path: str) -> PreTrainedModel:
        """Load the model from path."""
        try:
            if self.model_type.lower() == "lstm":
                config = LSTMConfig.from_pretrained(model_path)
                model = LSTMForLanguageModeling.from_pretrained(model_path, config=config)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path)
            print("Model has been loaded")
            return model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    def add_special_tokens(self, special_token_lst: list[str] = ["'", "|"]):
        for special_token in special_token_lst:
            self.tokenizer.add_tokens(special_token)
        print(f"Added {len(special_token_lst)} special tokens to the tokenizer")

    def generate_text(self, word_num: int, temp_lst: list[float]) -> dict[str, str]:
        """Generate text with different temperatures."""
        try:
            results = {}
            random_token_id = random.randint(0, 25)

            for temp in temp_lst:
                if self.use_vllm:
                    # vLLM generation path - handles multi-GPU automatically
                    sampling_params = SamplingParams(
                        temperature=temp,
                        max_tokens=word_num,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                    )
                    outputs = self.model.generate(self.tokenizer.decode([random_token_id]), sampling_params)
                    if not outputs or not outputs[0].outputs:
                        raise ValueError("vLLM generated empty output")
                    gen = outputs[0].outputs[0].text

                else:
                    # Traditional generation path
                    with torch.no_grad():
                        # Ensure input_ids is on the correct device
                        input_ids = torch.tensor([[random_token_id]], device=self.device)
                        gen = self.tokenizer.decode([random_token_id])
                        bar_count = 0

                        while bar_count < word_num and input_ids.shape[1] < self.max_length:
                            try:
                                curr_length = input_ids.shape[1]
                                position_ids = torch.arange(curr_length, device=self.device).unsqueeze(0)

                                outputs = self.model.generate(
                                    input_ids=input_ids,
                                    max_length=curr_length + 1,
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

                                # Reset context window if near max length
                                if input_ids.shape[1] >= self.max_length - 2:
                                    input_ids = input_ids[:, :1]

                            except RuntimeError as e:
                                if "out of memory" in str(e):
                                    torch.cuda.empty_cache()
                                    if hasattr(self.model, "module"):
                                        self.model.module.zero_grad(set_to_none=True)
                                    else:
                                        self.model.zero_grad(set_to_none=True)
                                    continue
                                raise

                results[f"unprompted_{temp}"] = gen

            return results

        except Exception as e:
            logging.error(f"Fatal error in generate_text: {str(e)}")
            torch.cuda.empty_cache()
            if hasattr(self.model, "module"):
                self.model.module.zero_grad(set_to_none=True)
            else:
                self.model.zero_grad(set_to_none=True)
            raise RuntimeError(f"Text generation failed: {str(e)}")

        finally:
            torch.cuda.empty_cache()


class BatchProcessor:
    """Handles batch processing of text generation."""

    def __init__(self, generator: TextGenerator, save_path: Path, chunk_size: int = 10) -> None:
        """Initialize batch processor."""
        self.generator = generator
        self.save_path = Path(save_path)
        self.chunk_size = chunk_size
        self.logger = Logger.setup(self.save_path)

    def process_batch(self, batch: pd.DataFrame, temp_lst: list[float]) -> pd.DataFrame:
        """Process a batch of data for text generation."""
        try:
            # Setup temperature columns
            temp_columns = [f"unprompted_{temp}" for temp in temp_lst]
            results = []

            # Handle different model types
            # Get available GPUs for LSTM processing
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

                # Split batch for multi-GPU processing
            if num_gpus > 1:
                sub_batches = np.array_split(batch, num_gpus)
                self.logger.info(f"Split batch into {len(sub_batches)} sub-batches for {num_gpus} GPUs")

                    # Process each sub-batch with error handling
                for gpu_idx, sub_batch in enumerate(sub_batches):
                    try:
                        # Set device for this sub-batch and update generator's device
                        if torch.cuda.is_available():
                            torch.cuda.set_device(gpu_idx)
                            self.generator.device = f"cuda:{gpu_idx}"
                            # Move model to current GPU if not using DDP
                            if not isinstance(self.generator.model, torch.nn.parallel.DistributedDataParallel):
                                    self.generator.model = self.generator.model.to(self.generator.device)

                        with torch.cuda.amp.autocast():
                            sub_results = self._process_subbatch(
                                    sub_batch, temp_lst, temp_columns, device_idx=gpu_idx
                                )
                            results.extend(sub_results)

                    except Exception as e:
                        self.logger.error(f"Error processing sub-batch on GPU {gpu_idx}: {str(e)}")
                        empty_results = [pd.Series({col: "" for col in temp_columns})] * len(sub_batch)
                        results.extend(empty_results)

                    finally:
                            # Cleanup after each sub-batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize(gpu_idx)
            else:
                # Single GPU/CPU processing
                results.extend(self._process_subbatch(batch, temp_lst, temp_columns))

            # Convert results to DataFrame
            try:
                result_df = pd.DataFrame(results, index=batch.index)
                missing_cols = set(temp_columns) - set(result_df.columns)
                if missing_cols:
                    self.logger.warning(f"Missing columns in results: {missing_cols}")
                    for col in missing_cols:
                        result_df[col] = ""
                return result_df

            except Exception as e:
                self.logger.error(f"Error creating result DataFrame: {str(e)}")
                return pd.DataFrame(columns=temp_columns, index=batch.index)

        except Exception as e:
            self.logger.exception(f"Critical error in batch processing: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(self.generator.model, "module"):
                self.generator.model.module.zero_grad(set_to_none=True)
            else:
                self.generator.model.zero_grad(set_to_none=True)
            raise RuntimeError(f"Batch processing failed: {str(e)}")

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _process_subbatch(
        self, sub_batch: pd.DataFrame, temp_lst: list[float], temp_columns: list[str], device_idx: int = 0
    ) -> list:
        """Process a sub-batch of data."""
        results = []
        for _, row in sub_batch.iterrows():
            try:
                # Generate text for this row
                result = self.generator.generate_text(row["sent_len"], temp_lst)
                results.append(pd.Series(result))
            except Exception as e:
                self.logger.error(f"Error processing row on device {device_idx}: {str(e)}")
                # Add empty result for failed row
                results.append(pd.Series({col: "" for col in temp_columns}))
            finally:
                # Cleanup after each row
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return results

    def segment_df(self, source_df: pd.DataFrame, ref_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Select rows to be generated and match source/ref dataframes."""
        self.logger.info(f"Segmenting DataFrames - source shape: {source_df.shape}, ref shape: {ref_df.shape}")

        # Keep all metadata columns up to and including 'model'
        info_cols = ref_df.columns[: ref_df.columns.get_loc("model") + 1].tolist()
        # Get generation result columns after 'model'
        gen_cols = source_df.columns[source_df.columns.get_loc("model") + 1 :].tolist()

        self.logger.info(f"Info columns: {info_cols}")
        self.logger.info(f"Generation columns: {gen_cols}")

        generated_df = pd.DataFrame()
        remaining_df = pd.DataFrame()

        # Process each sentence length group
        total_processed = 0
        total_remaining = 0

        for sent_len, ref_group in ref_df.groupby("sent_len"):
            # Find matching rows in source
            source_matches = source_df[source_df["sent_len"] == sent_len]
            match_count = min(len(source_matches), len(ref_group))

            self.logger.info(
                f"Length {sent_len}: {len(ref_group)} reference rows, {len(source_matches)} existing generations"
            )

            if match_count > 0:
                # Combine metadata with existing generations
                matches_df = pd.concat(
                    [ref_group.iloc[:match_count][info_cols], source_matches.iloc[:match_count][gen_cols]], axis=1
                )
                generated_df = pd.concat([generated_df, matches_df])
                total_processed += match_count

            # Identify rows needing generation
            if len(ref_group) > match_count:
                new_rows = ref_group.iloc[match_count:].copy()
                remaining_df = pd.concat([remaining_df, new_rows])
                total_remaining += len(ref_group) - match_count

        self.logger.info(f"Total processed rows: {total_processed}, remaining rows: {total_remaining}")
        return generated_df, remaining_df

    def process_dataframe(self, df: pd.DataFrame, temp_lst: list[float], resume: bool = False) -> pd.DataFrame:
        """Process entire dataframe with intermediate saves."""
        try:
            if self.generator.local_rank != -1:
                dist.barrier()

            # Track metadata columns
            info_cols = df.columns[: df.columns.get_loc("model") + 1].tolist()
            resume_file = self.save_path / "gen_intermediate.csv"

            # Handle resume logic
            if resume and resume_file.is_file():
                self.logger.info(f"Attempting to resume from {resume_file}")
                source_df = pd.read_csv(resume_file)
                generated_df, remaining_df = self.segment_df(source_df, df)
                self.logger.info(
                    f"Resume status: {len(generated_df)} rows recovered, {len(remaining_df)} rows remaining"
                )
                df = remaining_df  # Set remaining rows for processing
            else:
                self.logger.info("Starting fresh generation")
                generated_df = pd.DataFrame()

            # Process remaining rows if any
            total_rows = len(df)
            if total_rows > 0:
                self.logger.info(f"Processing {total_rows} rows in chunks of {self.chunk_size}")
                chunks = [df.iloc[i : i + self.chunk_size] for i in range(0, total_rows, self.chunk_size)]

                # Process each chunk
                for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
                    if self.generator.local_rank != -1:
                        dist.barrier()

                    # Process batches within chunk
                    processed_chunks = []
                    for i in range(0, len(chunk), self.chunk_size):
                        batch = chunk.iloc[i : i + self.chunk_size].copy()
                        processed_batch = self.process_batch(batch, temp_lst)

                        # Preserve metadata columns
                        for col in info_cols:
                            processed_batch[col] = batch[col]

                        processed_chunks.append(processed_batch)
                        torch.cuda.empty_cache()

                    # Combine chunk results
                    processed_df = pd.concat(processed_chunks)
                    generated_df = pd.concat([generated_df, processed_df])

                    # Save intermediate results if resuming
                    if resume and resume_file.is_file():
                        generated_df.to_csv(resume_file, index=False)
                        self.logger.info(f"Saved intermediate results - Total rows processed: {len(generated_df)}")
            else:
                self.logger.info("No new rows to process")

            # Ensure consistent column ordering
            if not generated_df.empty:
                all_cols = info_cols + [col for col in generated_df.columns if col not in info_cols]
                generated_df = generated_df[all_cols]

            return generated_df

        except Exception as e:
            self.logger.error(f"Error in process_dataframe: {str(e)}")
            raise


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
        try:
            # Choose generation strategy
            if do_sample:
                return self._generate_with_sampling(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                )
            return self._generate_greedy(
                input_ids=input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
            )

        except Exception as e:
            logging.error(f"Generation failed: {str(e)}")
            torch.cuda.empty_cache()
            raise

    def _generate_greedy(
        self,
        input_ids: torch.LongTensor,
        max_length: int,
        num_return_sequences: int = 1,
    ) -> torch.LongTensor:
        """Generate sequences using greedy decoding."""
        if num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)

        generated = input_ids.clone()
        try:
            with torch.no_grad():
                for _ in range(max_length - input_ids.shape[1]):
                    outputs = self(input_ids=generated)
                    next_token_logits = outputs["logits"][:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
        except Exception as e:
            logging.error(f"Error in greedy generation: {str(e)}")
            raise

        return generated

    def _generate_with_sampling(
        self,
        input_ids: torch.LongTensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
    ) -> torch.LongTensor:
        """Generate sequences using sampling with temperature, top-k, and top-p filtering."""
        if num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)

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
            logging.error(f"Error in sampling generation: {str(e)}")
            raise

        return generated
