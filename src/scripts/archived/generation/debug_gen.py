import logging
import string
from pathlib import  Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, PreTrainedModel

from lexical_benchmark.utils import hf_util


class TextGenerator:
    """Handles text generation using transformer models."""

    def __init__(
        self,
        model_path: str,
        model_max_length: int = 1024,
        model_type: str = "transformer",
        local_rank: int = -1,
        max_word_len: int = 2,
    ):

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
        self.tokenizer = hf_util.CharacterTokenizer(chars=chars, model_max_length=model_max_length)

        # Model initialization with distributed support
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
            model = AutoModelForCausalLM.from_pretrained(model_path)
            print(f"{self.model_type} Model has been loaded")
            return model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    def add_special_tokens(self, special_token_lst: list[str] = ["'", "|"]):
        for special_token in special_token_lst:
            self.tokenizer.add_tokens(special_token)
        print(f"Added {len(special_token_lst)} special tokens to the tokenizer: {special_token_lst}")

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
            raise RuntimeError(f"Vanilla token generation failed: {str(e)}")

    def _handle_oom(self):
        """Handle out of memory errors."""
        torch.cuda.empty_cache()
        # Traditional model cleanup
        if hasattr(self.model, "module"):
            self.model.module.zero_grad(set_to_none=True)
        else:
            self.model.zero_grad(set_to_none=True)

    def _initialize_generation(self):
        """Initialize generation parameters."""
        random_token_id = self.tokenizer.get_id("|")
        input_ids = torch.tensor([[random_token_id]], device=self.device)
        gen = self.tokenizer.decode([random_token_id])
        return input_ids, gen

    def _should_reset_context(self, input_ids):
        """Check if context window should be reset."""
        return input_ids.shape[1] >= self.max_length - 2

    def _reset_context(self, input_ids):
        """Reset context window."""
        return input_ids[:, :1]

    def generate_text(self, word_num: int, temp_lst: list[float]) -> dict[str, str]:
        """Generate text with different temperatures."""
        try:
            results = {}
            max_retries = 5

            for temp in temp_lst:
                input_ids, gen = self._initialize_generation()
                bar_count = 0
                retry_count = 0
                cur_word_len = 0
                while bar_count < word_num:
                    if self._should_reset_context(input_ids):
                        input_ids = self._reset_context(input_ids)
                        continue

                    try:
                        with torch.no_grad():
                                new_token, outputs = self.generate_next_token_vanilla(input_ids, temp)
                                if new_token is None:
                                    retry_count += 1
                                    if retry_count >= max_retries:
                                        raise RuntimeError("Maximum retries exceeded for OOM recovery")
                                    continue
                                decoded_token = self.tokenizer.decode([new_token])
                                input_ids = outputs

                        # Handle consecutive bars
                        if decoded_token == "|" and gen[-1] == "|":
                            continue  # Skip this token and try again

                        # Handle long words before adding new token
                        if cur_word_len >= self.max_word_len and decoded_token != "|":
                            gen += "|"
                            bar_count += 1
                            cur_word_len = 0
                            print(f"Add word boundary to {gen} with {len(gen)=}")


                        # Update generation
                        gen += decoded_token
                        # Update counters
                        if decoded_token == "|":
                            bar_count += 1
                            cur_word_len = 0
                        else:
                            cur_word_len += 1

                        retry_count = 0

                    except Exception as e:
                        logging.warning(f"Error during token generation: {str(e)}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise RuntimeError(f"Maximum retries exceeded: {str(e)}")
                        continue

                results[f"unprompted_{temp}"] = gen

            return results

        except Exception as e:
            logging.error(f"Fatal error in generate_text: {str(e)}")
            self._handle_oom()
            raise RuntimeError(f"Text generation failed: {str(e)}")

        finally:
            torch.cuda.empty_cache()

MODEL_PATH = Path("/lustre/fswork/projects/rech/hhb/ucx81cx/data/models") / "ChildRealistic/by_month/EN/30/00/trans"
text_gen = TextGenerator(
    model_path=str(MODEL_PATH),
    max_word_len=50,
)
word_num = 2
temp_lst = [0.3,0.6,1.0,1.5]
# text_gen.generate_text()
