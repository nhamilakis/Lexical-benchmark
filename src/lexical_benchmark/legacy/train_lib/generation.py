import logging
import random
import string
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from vllm import LLM, SamplingParams

from lexical_benchmark.train_lib import hf_tools


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

    @staticmethod
    def setup_stdout(*, debug: bool = False) -> logging.Logger:
        """Configure STDOUT logs."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.DEBUG if debug else logging.INFO,
            handlers=[logging.StreamHandler()],
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
        max_word_len: int = 50,
    ):
        # Prevent vLLM for LSTM
        self.use_vllm = use_vllm and model_type.lower() != "lstm"
        self.model_type = model_type
        self.local_rank = local_rank
        self.max_word_len = max_word_len
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
        self.tokenizer = hf_tools.CharacterTokenizer(chars=chars, model_max_length=model_max_length)

        # Model initialization with distributed support
        if self.use_vllm:
            # For vLLM, we don't need to handle device placement manually
            tensor_parallel_size = torch.cuda.device_count() if local_rank != -1 else 1
            self.model = LLM(
                model=model_path,
                skip_tokenizer_init=True,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=0.9,
            )

        else:
            self.model = self._load_model(model_path)
            if local_rank != -1:
                self.model = DistributedDataParallel(self.model, device_ids=[local_rank], find_unused_parameters=True)
            self.model.eval()

        if local_rank != -1:
            dist.barrier()  # Sync after initialization

        self.max_length = model_max_length

    def _load_model(self, model_path: str) -> PreTrainedModel:
        """Load the model from path."""
        try:
            if self.model_type.lower() == "lstm":
                config = LSTMConfig.from_pretrained(model_path)
                model = LSTMForLanguageModeling.from_pretrained(model_path, config=config)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path)
            print(f"{self.model_type} Model has been loaded")
            return model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    def add_special_tokens(self, special_token_lst: list[str] = ["'", "|"]):
        for special_token in special_token_lst:
            self.tokenizer.add_tokens(special_token)
        print(f"Added {len(special_token_lst)} special tokens to the tokenizer: {special_token_lst}")

    def generate_next_token_vllm(self, input_ids, sampling_params):
        """Generate next token using vLLM."""
        try:
            if not isinstance(input_ids, list):
                input_ids = list(input_ids)
            # NOTE: deprecation does not seem to be active, prompt_token_ids will be removed
            # NOTE: and integrated into prompts ==> keep this in mind
            outputs = self.model.generate(
                prompt_token_ids=input_ids,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            if not outputs or not outputs[0].outputs:
                raise ValueError("vLLM generated empty output")

            token_ids = list(outputs[0].outputs[0].token_ids)
            return token_ids, outputs

        except RuntimeError as e:
            if "out of memory" in str(e):
                self._handle_oom()
                return None, None
            raise
        except Exception as e:
            raise RuntimeError(f"vLLM token generation failed: {e!s}")

    def generate_next_token_vanilla(self, input_ids, temperature):
        """Generate next token using vanilla generation."""
        try:
            curr_length = input_ids.shape[1]
            position_ids = torch.arange(curr_length, device=self.device).unsqueeze(0)

            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=curr_length + 1,
                do_sample=True,
                temperature=temperature,
                num_beams=1,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                position_ids=position_ids,
                use_cache=True,
            )

            new_token = outputs[0, -1].item()
            return new_token, outputs
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._handle_oom()
                return None, None
            raise
        except Exception as e:
            raise RuntimeError(f"Vanilla token generation failed: {e!s}")

    def _handle_oom(self):
        """Handle out of memory errors."""
        torch.cuda.empty_cache()
        if self.use_vllm:
            # For vLLM, we might need additional cleanup
            if hasattr(self.model, "engine"):
                self.model.engine.empty_cache()
        # Traditional model cleanup
        elif hasattr(self.model, "module"):
            self.model.module.zero_grad(set_to_none=True)
        else:
            self.model.zero_grad(set_to_none=True)

    def _initialize_generation(self):
        """Initialize generation parameters."""
        random_token_id = self.tokenizer.get_id("|")
        if self.use_vllm:
            input_ids = [random_token_id]
            gen = self.tokenizer.decode([random_token_id])
        else:
            input_ids = torch.tensor([[random_token_id]], device=self.device)
            gen = self.tokenizer.decode([random_token_id])
        return input_ids, gen

    def _should_reset_context(self, input_ids):
        """Check if context window should be reset."""
        if self.use_vllm:
            return len(input_ids) >= self.max_length - 2
        return input_ids.shape[1] >= self.max_length - 2

    def _reset_context(self, input_ids):
        """Reset context window."""
        if self.use_vllm:
            return input_ids[:-1]
        return input_ids[:, :1]

    def _handle_post_generation(
        self,
        decoded_token: str,
        gen: str,
        consecutive_bars: int,
        cur_word_len: int,
        bar_count: int,
        random_range: list[int],
        input_ids: torch.Tensor | list,
        is_vllm: bool,
        max_retries: int = 3,
    ) -> tuple[str, str, int, int, int, torch.Tensor | list, bool]:
        """Handle post-generation conditions and updates."""
        should_retry = False

        # First handle overly-long words
        if cur_word_len > self.max_word_len and decoded_token != "|":
            if not gen.endswith("|"):
                decoded_token = "|"
                bar_token_id = self.tokenizer.encode("|")[0]
                if is_vllm:
                    input_ids.extend([bar_token_id])
                else:
                    new_token = torch.tensor([[bar_token_id]], device=input_ids.device)
                    input_ids = torch.cat([input_ids, new_token], dim=1)
                bar_count += 1
                cur_word_len = 0
            else:
                should_retry = True
                return gen, "", consecutive_bars, cur_word_len, bar_count, input_ids, should_retry

        # Handle consecutive bars with retries
        if decoded_token == "|":
            consecutive_bars += 1
            if gen.endswith("|"):  # Would create consecutive bars
                if consecutive_bars < (2 + max_retries):
                    # Still have retries left, signal for retry
                    should_retry = True
                    return gen, "", consecutive_bars, cur_word_len, bar_count, input_ids, should_retry
                # Exhausted retries, use random token
                print("Exhausted retries, use random token")
                random_token_id = random.randint(random_range[0], random_range[1])
                decoded_token = self.tokenizer.decode([random_token_id])
                if is_vllm:
                    input_ids.extend([random_token_id])
                else:
                    new_token = torch.tensor([[random_token_id]], device=input_ids.device)
                    input_ids = torch.cat([input_ids, new_token], dim=1)
                consecutive_bars = 0
                cur_word_len = 1  # Start counting new word
        else:
            consecutive_bars = 0

        # Update generation
        gen += decoded_token

        # Update counters for next iteration
        if decoded_token == "|":
            bar_count += 1
            cur_word_len = 0
        else:
            cur_word_len += 1

        return gen, decoded_token, consecutive_bars, cur_word_len, bar_count, input_ids, should_retry

    def generate_text(self, word_num: int, temp_lst: list[float], random_range=[0, 25]) -> dict[str, str]:
        """Generate text with different temperatures."""
        try:
            results = {}
            max_retries = 10
            for temp in temp_lst:
                input_ids, gen = self._initialize_generation()
                bar_count = 0
                retry_count = 0
                cur_word_len = 0
                consecutive_bars = 0

                if self.use_vllm:
                    sampling_params = SamplingParams(
                        temperature=temp,
                        max_tokens=word_num,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                    )

                while bar_count < word_num:
                    if self._should_reset_context(input_ids):
                        input_ids = self._reset_context(input_ids)
                        continue

                    try:
                        # Generate next token
                        if self.use_vllm:
                            token_ids, outputs = self.generate_next_token_vllm(input_ids, sampling_params)
                            if token_ids is None:
                                retry_count += 1
                                if retry_count >= max_retries:
                                    raise RuntimeError("Maximum retries exceeded for OOM recovery")
                                continue
                            decoded_token = self.tokenizer.decode([token_ids[-1]])
                            if token_ids is not None:
                                input_ids.extend(token_ids)
                        else:
                            with torch.no_grad():
                                new_token, outputs = self.generate_next_token_vanilla(input_ids, temp)
                                if new_token is None:
                                    retry_count += 1
                                    if retry_count >= max_retries:
                                        raise RuntimeError("Maximum retries exceeded for OOM recovery")
                                    continue
                                decoded_token = self.tokenizer.decode([new_token])
                                input_ids = outputs

                        # Handle post-generation conditions
                        gen, decoded_token, consecutive_bars, cur_word_len, bar_count, input_ids, should_retry = (
                            self._handle_post_generation(
                                decoded_token,
                                gen,
                                consecutive_bars,
                                cur_word_len,
                                bar_count,
                                random_range,
                                input_ids,
                                self.use_vllm,
                            )
                        )

                        if should_retry:
                            continue  # Skip this token and generate a new one

                        retry_count = 0

                    except Exception as e:
                        logging.warning(f"Error during token generation: {e!s}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise RuntimeError(f"Maximum retries exceeded: {e!s}")
                        continue

                results[f"unprompted_{temp}"] = gen
            return results

        except Exception as e:
            logging.exception(f"Fatal error in generate_text: {e!s}")
            self._handle_oom()
            raise RuntimeError(f"Text generation failed: {e!s}")
        finally:
            torch.cuda.empty_cache()
            if self.use_vllm and hasattr(self.model, "engine"):
                self.model.engine.empty_cache()


class BatchProcessor:
    """Handles batch processing of text generation."""

    def __init__(
        self, generator: TextGenerator, save_path: Path, hour_per_year: int, chunk_size: int = 10, debug: bool = False
    ) -> None:
        """Initialize batch processor."""
        self.hour_per_year = hour_per_year
        self.generator = generator
        self.save_path = Path(save_path)
        self.chunk_size = chunk_size
        self.logger = Logger.setup_stdout(debug=debug)
        self.debug = debug

    def get_save_file(self, *, intermidiate: bool = False, debug: bool = False) -> Path:
        """Build target file."""
        file_name = f"{self.hour_per_year}_hour_per_year"

        if intermidiate:
            file_name = f"{file_name}.intermediate"

        if debug:
            file_name = f"{file_name}.debug"

        return self.save_path / f"{file_name}.csv"

    def process_batch(self, batch: pd.DataFrame, temp_lst: list[float]) -> pd.DataFrame:
        """Process a batch of data for text generation."""
        try:
            # Setup temperature columns
            temp_columns = [f"unprompted_{temp}" for temp in temp_lst]
            results = []

            if self.generator.use_vllm:
                # For vLLM, we don't split batches or manage devices manually
                self.logger.info("Using vLLM for generation - processing full batch")
                try:
                    results.extend(self._process_subbatch(batch, temp_lst, temp_columns))
                except Exception as e:
                    self.logger.error(f"Error in vLLM batch processing: {e!s}")
                    empty_results = [pd.Series({col: "" for col in temp_columns})] * len(batch)
                    results.extend(empty_results)
            else:
                num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
                if num_gpus > 1:
                    # Split batch only for traditional models
                    sub_batches = np.array_split(batch, num_gpus)
                    self.logger.info(f"Split batch into {len(sub_batches)} sub-batches for {num_gpus} GPUs")

                    for gpu_idx, sub_batch in enumerate(sub_batches):
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.set_device(gpu_idx)
                                self.generator.device = f"cuda:{gpu_idx}"
                                # Move model to device only for traditional models
                                if not isinstance(self.generator.model, torch.nn.parallel.DistributedDataParallel):
                                    self.generator.model = self.generator.model.to(self.generator.device)

                            with torch.cuda.amp.autocast():
                                sub_results = self._process_subbatch(
                                    sub_batch, temp_lst, temp_columns, device_idx=gpu_idx
                                )
                                results.extend(sub_results)

                        except Exception as e:
                            self.logger.error(f"Error processing sub-batch on GPU {gpu_idx}: {e!s}")
                            empty_results = [pd.Series({col: "" for col in temp_columns})] * len(sub_batch)
                            results.extend(empty_results)

                        finally:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize(gpu_idx)
                else:
                    # Single GPU/CPU processing
                    results.extend(self._process_subbatch(batch, temp_lst, temp_columns))

            try:
                result_df = pd.DataFrame(results, index=batch.index)
                missing_cols = set(temp_columns) - set(result_df.columns)
                if missing_cols:
                    self.logger.warning(f"Missing columns in results: {missing_cols}")
                    for col in missing_cols:
                        result_df[col] = ""
                return result_df

            except Exception as e:
                self.logger.error(f"Error creating result DataFrame: {e!s}")
                return pd.DataFrame(columns=temp_columns, index=batch.index)

        except Exception as e:
            self.logger.exception(f"Critical error in batch processing: {e!s}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not self.generator.use_vllm:  # Only for traditional models
                if hasattr(self.generator.model, "module"):
                    self.generator.model.module.zero_grad(set_to_none=True)
                else:
                    self.generator.model.zero_grad(set_to_none=True)
            raise RuntimeError(f"Batch processing failed: {e!s}")

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
                self.logger.error(f"Error processing row on device {device_idx}: {e!s}")
                # Add empty result for failed row
                results.append(pd.Series({col: "" for col in temp_columns}))
            finally:
                # Cleanup after each row
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return results

    def process_dataframe(
        self, prompt_df: pd.DataFrame, temp_lst: list[float], resume: bool = False
    ) -> pd.DataFrame | None:
        """Process entire dataframe with intermediate saves."""
        if self.generator.local_rank != -1:
            dist.barrier()

        # Track metadata columns
        info_cols = prompt_df.columns[: prompt_df.columns.get_loc("model") + 1].tolist()
        resume_file = self.get_save_file(intermidiate=True)

        # Handle resume logic
        if resume and resume_file.is_file():
            self.logger.info(f"Attempting to resume from {resume_file}")
            generated_df = pd.read_csv(resume_file)
            self.logger.info(f"Resume status: {len(generated_df)} rows recovered.")
        else:
            self.logger.info("Starting fresh generation")
            generated_df = pd.DataFrame()

        current_line = len(generated_df)
        # Move prompt to a last generated line
        prompt_df = prompt_df.iloc[current_line:]

        # Process remaining rows if any
        total_rows = len(prompt_df)
        if total_rows <= 0:
            self.logger.info("No new rows to process")
            return None

        self.logger.info(f"Processing {total_rows} rows in chunks of {self.chunk_size}")

        # Process each chunk
        self.logger.info("Starting generation...")
        for current_index in range(0, len(prompt_df), self.chunk_size):
            current_batch = prompt_df.iloc[current_index : current_index + self.chunk_size]
            processed_batch = self.process_batch(current_batch, temp_lst)

            # Preserve metadata columns
            for col in info_cols:
                processed_batch[col] = current_batch[col]
            torch.cuda.empty_cache()

            # Append Generation
            generated_df = pd.concat([generated_df, processed_batch])

            # Save intermediate results if resuminggit
            generated_df.to_csv(self.get_save_file(intermidiate=True, debug=self.debug))
            self.logger.info(f"Saved intermediate results - Total rows processed: {len(generated_df)}")

        # Ensure consistent column ordering
        if not generated_df.empty:
            all_cols = info_cols + [col for col in generated_df.columns if col not in info_cols]
            generated_df = generated_df[all_cols]

        return generated_df


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
            logging.exception(f"Generation failed: {e!s}")
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
            logging.exception(f"Error in greedy generation: {e!s}")
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
            logging.exception(f"Error in sampling generation: {e!s}")
            raise

        return generated
