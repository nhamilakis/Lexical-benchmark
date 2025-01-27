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
            self.model = LLM(model=model_path)
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
            # Set device for distributed training
            if self.local_rank != -1:
                torch.cuda.set_device(self.local_rank)

            # Sync before vLLM generation in distributed setting
            if self.use_vllm and self.local_rank != -1:
                dist.barrier()

            results = {}
            random_token_id = random.randint(0, 25)

            for temp in temp_lst:
                try:
                    if self.use_vllm:
                        # vLLM generation path
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
                                        use_cache=True,  # kv_cache to save time
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
                                        # Handle OOM by clearing cache and reducing batch
                                        torch.cuda.empty_cache()
                                        if hasattr(self.model, "module"):
                                            self.model.module.zero_grad(set_to_none=True)
                                        else:
                                            self.model.zero_grad(set_to_none=True)
                                        continue
                                    raise

                    results[f"unprompted_{temp}"] = gen

                except Exception as e:
                    # Log error for specific temperature but continue with others
                    logging.error(f"Generation failed for temperature {temp}: {str(e)}")
                    results[f"unprompted_{temp}"] = ""
                    continue

            return results

        except Exception as e:
            # Handle any unexpected errors
            logging.error(f"Fatal error in generate_text: {str(e)}")
            torch.cuda.empty_cache()
            if hasattr(self.model, "module"):
                self.model.module.zero_grad(set_to_none=True)
            else:
                self.model.zero_grad(set_to_none=True)
            raise RuntimeError(f"Text generation failed: {str(e)}")

        finally:
            # Ensure cleanup happens even if an error occurs
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
        # synchronization for multi-GPU
        if self.generator.local_rank != -1:
            dist.barrier()
        temp_columns = [f"unprompted_{temp}" for temp in temp_lst]
        results = []

        if isinstance(self.generator.model, LSTMForLanguageModeling):
            num_gpus = torch.cuda.device_count()
            sub_batches = np.array_split(batch, num_gpus) if num_gpus > 1 else [batch]

            for sub_batch in sub_batches:
                results.extend(self._process_subbatch(sub_batch, temp_lst, temp_columns))
        else:
            results.extend(self._process_subbatch(batch, temp_lst, temp_columns))

        batch[temp_columns] = pd.DataFrame(results, index=batch.index)
        return batch


    def process_batch(self, batch: pd.DataFrame, temp_lst: list[float]) -> pd.DataFrame:
        """Process a batch of data for text generation."""
        try:
            # Initial synchronization for multi-GPU
            if self.generator.local_rank != -1:
                dist.barrier()

            # Setup temperature columns
            temp_columns = [f"unprompted_{temp}" for temp in temp_lst]
            results = []

            # Handle different model types
            if isinstance(self.generator.model, LSTMForLanguageModeling):
                # Get available GPUs for LSTM processing
                num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
                # Split batch for multi-GPU processing
                if num_gpus > 1:
                    sub_batches = np.array_split(batch, num_gpus)
                    self.logger.info(f"Split batch into {len(sub_batches)} sub-batches for {num_gpus} GPUs")
                    # Process each sub-batch with error handling
                    for gpu_idx, sub_batch in enumerate(sub_batches):
                        try:
                            # Set device for this sub-batch
                            if torch.cuda.is_available():
                                torch.cuda.set_device(gpu_idx)
                            with torch.cuda.amp.autocast():  # Use mixed precision for efficiency
                                sub_results = self._process_subbatch(
                                    sub_batch, 
                                    temp_lst, 
                                    temp_columns,
                                    device_idx=gpu_idx
                                )
                                results.extend(sub_results)
                        except Exception as e:
                            self.logger.error(f"Error processing sub-batch on GPU {gpu_idx}: {str(e)}")
                            # Add empty results for failed sub-batch
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
            else:
                # Non-LSTM model processing (e.g., Transformer, vLLM)
                results.extend(self._process_subbatch(batch, temp_lst, temp_columns))

            # Convert results to DataFrame format
            try:
                result_df = pd.DataFrame(results, index=batch.index)
                # Verify all expected columns are present
                missing_cols = set(temp_columns) - set(result_df.columns)
                if missing_cols:
                    self.logger.warning(f"Missing columns in results: {missing_cols}")
                    for col in missing_cols:
                        result_df[col] = ""
                return result_df
            except Exception as e:
                self.logger.error(f"Error creating result DataFrame: {str(e)}")
                # Return empty DataFrame with correct structure
                return pd.DataFrame(columns=temp_columns, index=batch.index)

        except Exception as e:
            self.logger.exception(f"Critical error in batch processing: {str(e)}")
            # Cleanup in case of critical failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(self.generator.model, 'module'):
                self.generator.model.module.zero_grad(set_to_none=True)
            else:
                self.generator.model.zero_grad(set_to_none=True)
            raise RuntimeError(f"Batch processing failed: {str(e)}")

        finally:
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if self.generator.local_rank != -1:
                    torch.cuda.synchronize()

    def _process_subbatch(
        self,
        sub_batch: pd.DataFrame,
        temp_lst: list[float],
        temp_columns: list[str],
        device_idx: int = 0
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


    def segment_df(self, source_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
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
        if self.generator.local_rank != -1:
            dist.barrier()  # Sync at start

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
            if self.generator.local_rank != -1:
                dist.barrier()  # Sync before each chunk
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
        """Generate text tokens using the LSTM model with proper distributed handling."""
        try:
            if dist.is_initialized():
                return self._generate_distributed(
                    input_ids=input_ids,
                    max_length=max_length,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                )

            # Single GPU or CPU generation
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

    def _generate_distributed(
        self,
        input_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        num_return_sequences: int,
    ) -> torch.LongTensor:
        """Handle generation in distributed setting.

        This method properly distributes the generation workload across available GPUs.
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        try:
            # Calculate local batch size
            total_sequences = input_ids.size(0) * num_return_sequences
            base_sequence_per_gpu = total_sequences // world_size
            extra_sequences = total_sequences % world_size

            # Distribute any extra sequences
            local_sequences = base_sequence_per_gpu + (1 if rank < extra_sequences else 0)
            start_idx = rank * base_sequence_per_gpu + min(rank, extra_sequences)
            end_idx = start_idx + local_sequences

            # Process local chunk
            local_input_ids = input_ids[start_idx:end_idx]

            # Generate on local chunk
            if do_sample:
                local_output = self._generate_with_sampling(
                    input_ids=local_input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                )
            else:
                local_output = self._generate_greedy(
                    input_ids=local_input_ids,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                )

            # Gather results from all processes
            gathered_sizes = [torch.tensor(0, device=local_output.device) for _ in range(world_size)]
            local_size = torch.tensor(local_output.size(0), device=local_output.device)
            dist.all_gather(gathered_sizes, local_size)

            max_size = max(gathered_sizes).item()
            padded_local_output = torch.nn.functional.pad(
                local_output, (0, 0, 0, max_size - local_output.size(0)), value=self.config.pad_token_id
            )

            gathered_outputs = [torch.zeros_like(padded_local_output) for _ in range(world_size)]
            dist.all_gather(gathered_outputs, padded_local_output)

            # Remove padding and concatenate results
            final_outputs = []
            for output, size in zip(gathered_outputs, gathered_sizes):
                final_outputs.append(output[:size])

            return torch.cat(final_outputs)

        except Exception as e:
            logging.exception(f"Distributed generation failed on rank {rank}: {str(e)}")
            # Ensure all processes fail together
            dist.barrier()
            raise RuntimeError(f"Distributed generation failed on rank {rank}")

        finally:
            # Cleanup
            torch.cuda.empty_cache()

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

        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                outputs = self(input_ids=generated)
                next_token_logits = outputs["logits"][:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                # Early stopping if all sequences have hit the EOS token
                if (next_token == self.config.eos_token_id).all():
                    break

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

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                # Early stopping if all sequences have hit the EOS token
                if (next_token == self.config.eos_token_id).all():
                    break

        return generated
