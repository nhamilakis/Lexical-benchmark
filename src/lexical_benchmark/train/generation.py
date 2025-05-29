import logging
import string
import typing as t
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import vllm  # type: ignore[missing-dependency]
    from vllm import LLM, SamplingParams  # type: ignore[missing-dependency]
except ImportError:
    warnings.warn("'vllm' could not be imported, is it not installed ?", category=ImportWarning, stacklevel=1)
    vllm = None
    LLM, SamplingParams = (None, None)

from lexical_benchmark import lb_types

from .checkpoint_utils import ESTIMATION_MONTH_KEY_TYPE, GenerationCheckpoint, GenerationsStruct
from .trainers.lstm import LSTMConfig, LSTMForLanguageModeling

Model = t.Any

L = logging.getLogger(__name__)


class BatchGenerator:
    """Generator class with early stopping for long sequences."""

    def __init__(
        self,
        *,
        model_path: Path,
        tokenizer_name: str,
        device: lb_types.DEVICE_TYPE,
        use_vllm: bool,
        model_type: lb_types.MODEL_TYPE,
        batch_size: int = 1,
        max_word_length: int = 30,
        max_generation_length: int = 1024,
    ) -> None:
        self.device = device
        self.use_vllm = use_vllm
        self.model_type = model_type
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_word_length = max_word_length
        self.max_generation_length = max_generation_length

        # Set model path
        if model_path.is_dir():
            self.model_path = model_path
        elif model_path.is_file():
            self.model_path = model_path.parent
        else:
            raise ValueError(f"Given {model_path} does not exist !!")

        if not self.use_vllm:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = self.load_model()

    def load_model(self) -> Model:
        """Load the model."""
        if self.model_type == "lstm":
            config = LSTMConfig.from_pretrained(self.model_path)
            return LSTMForLanguageModeling.from_pretrained(self.model_path, config=config).to(self.device)
        if self.model_type == "gpt2":
            if self.use_vllm and vllm:
                return LLM(
                    model=str(self.model_path),
                    tokenizer=self.tokenizer_name,
                    gpu_memory_utilization=0.9,
                    tensor_parallel_size=1,
                )
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            return model.to(self.device)
        return None

    def _get_current_word_length(self, text: str) -> int:
        """Get length of current word (text after last boundary: | or punctuation)."""
        if not text:
            return 0

        # Find the last boundary (| or punctuation), handling consecutive boundaries
        last_boundary_pos = -1
        for i in range(len(text) - 1, -1, -1):
            char = text[i]
            if char == "|" or char in string.punctuation:
                last_boundary_pos = i
                break

        if last_boundary_pos == -1:
            # No boundary found, entire text is current word
            return len(text)

        # Skip consecutive boundaries to find actual word start
        word_start = last_boundary_pos + 1
        while word_start < len(text) and (text[word_start] == "|" or text[word_start] in string.punctuation):
            word_start += 1

        # Return length from word start to end
        return len(text) - word_start

    def _generate_sequence(self, temperature) -> str:
        """Generate sequences with early stopping."""
        if self.use_vllm and vllm:
            return self._generate_vllm(temperature)
        return self._generate_batch_vanilla(temperature)

    def _generate_batch_vanilla(self, temperature: float) -> str:
        """Generate with early stopping using transformers."""
        input_ids = self.tokenizer("", return_tensors="pt").input_ids.to(self.device)
        generated_text = ""

        with torch.no_grad():
            for _ in range(self.max_generation_length):
                # Generate next token
                outputs = self.model(input_ids)

                # Handle both dict and object outputs
                if isinstance(outputs, dict):
                    # Custom LSTM model returns dict
                    logits = (
                        outputs["logits"][0, -1, :] if "logits" in outputs else outputs["prediction_scores"][0, -1, :]
                    )
                else:
                    # Standard transformers model returns object
                    logits = outputs.logits[0, -1, :]

                # Apply temperature sampling
                if temperature > 0:
                    logits = logits / temperature
                    probabilities = torch.softmax(logits, dim=-1)
                    next_token_id = torch.multinomial(probabilities, 1)
                else:
                    next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

                # Decode the new token
                next_token = self.tokenizer.decode(next_token_id, skip_special_tokens=True)

                # Check current word length after adding this token
                potential_text = generated_text + next_token
                current_word_length = self._get_current_word_length(potential_text)

                if current_word_length > self.max_word_length:
                    L.warning(f"Early stopping: word length {current_word_length} exceeds {self.max_word_length}")
                    # Force add boundary marker and stop
                    generated_text += "|"
                    break

                # Add the token and continue
                generated_text += next_token
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

                # Check for natural stopping (EOS token)
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        return generated_text

    def _generate_vllm(self, temperature: float) -> str:
        """Generate with early stopping using vLLM."""
        # Use shorter generation for efficiency
        adjusted_max_tokens = min(256, self.max_generation_length)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=adjusted_max_tokens,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        outputs = self.model.generate(
            prompts="",
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        raw_text = outputs[0].outputs[0].text

        # Apply early stopping: monitor word length during processing
        result = ""
        for char in raw_text:
            potential_result = result + char
            current_word_length = self._get_current_word_length(potential_result)

            if current_word_length > self.max_word_length:
                L.warning(f"Early stopping: word length {current_word_length} exceeds {self.max_word_length}")
                # Force boundary and stop
                result += "|"
                break

            result += char

        # Return original text without modification
        return result

    def save_generation(
        self,
        temperature: float,
        gen_attrs: dict[ESTIMATION_MONTH_KEY_TYPE, GenerationsStruct],
        target_dir: Path,
        *,
        resume: bool = True,
        override: bool = False,
    ) -> None:
        """Generate text from model."""
        resume_checkpoint = None

        if resume:
            resume_checkpoint = GenerationCheckpoint.load_intermediate(location=target_dir, temperature=temperature)
            if resume_checkpoint:
                L.info(f"Resuming generation from {target_dir / f'generation_{temperature}.intermediate.obj'}")

        if override or not resume_checkpoint:
            L.info("Starting generation...")
            resume_checkpoint = GenerationCheckpoint.init_from_args(
                temperature=temperature, word_counts=gen_attrs, location=target_dir
            )
            target_dir.mkdir(exist_ok=True, parents=True)

        if resume_checkpoint.remaining_count() == 0:
            L.info("No more items require generation, exiting")
            return

        while resume_checkpoint.remaining_count() > 0:
            next_id, leftover = resume_checkpoint.get_next_gen()
            resume_checkpoint = self.generate_checkpoint_text(
                temperature=temperature,
                nb_tokens=leftover,
                index=next_id,
                checkpoint=resume_checkpoint,
            )
            # save intermdeaite generation
            resume_checkpoint.save_intermediate(target_dir)
            L.info(f"Save intermediate checkpoint @ {target_dir}")

        resume_checkpoint.save_final(target_dir)
        L.info(f"Completed generation, checkpoint can be found @ {target_dir}")

    def generate_text(self, temperature: float, nb_tokens: int) -> tuple[str, int]:
        """Generate text from model."""
        generated_text = ""
        curr_tokens = 0

        while curr_tokens < nb_tokens:
            new_text = self._generate_sequence(temperature)
            generated_text += new_text
            curr_tokens = self._count_words(generated_text)
            if curr_tokens >= nb_tokens:
                break

        return generated_text, curr_tokens

    def generate_checkpoint_text(
        self, temperature: float, nb_tokens: int, index: ESTIMATION_MONTH_KEY_TYPE, checkpoint: GenerationCheckpoint
    ) -> GenerationCheckpoint:
        """Generate text from model, and save it into the checkpoint."""
        curr_tokens = 0

        while curr_tokens < nb_tokens:
            new_text = self._generate_sequence(temperature)
            count = self._count_words(new_text)
            checkpoint.append_text_list(index, [new_text], count)
            curr_tokens += count
            if curr_tokens >= nb_tokens:
                break
        return checkpoint

    def _count_words(self, generated_text: str) -> int:
        """Count words using | and punctuation as delimiters, handling consecutive boundaries."""
        if not generated_text.strip():
            return 0

        words = []
        current_word = ""

        for char in generated_text:
            if char == "|" or char in string.punctuation:
                # Hit a boundary
                if current_word.strip():  # If we have accumulated a word
                    words.append(current_word.strip())
                    current_word = ""
                # Skip consecutive boundaries (don't add empty words)
            else:
                # Regular character, add to current word
                current_word += char

        # Add final word if exists
        if current_word.strip():
            words.append(current_word.strip())

        return len(words)

    def _cut_text(self, generated_text: str, nb_tokens: int) -> str:
        """Cut text to specified number of tokens, handling consecutive boundaries properly."""
        if not generated_text.strip():
            return ""

        words = []
        current_word = ""
        word_boundaries = []  # Track where boundaries occur

        for i, char in enumerate(generated_text):
            if char == "|" or char in string.punctuation:
                # Hit a boundary
                if current_word.strip():  # If we have accumulated a word
                    words.append(current_word.strip())
                    word_boundaries.append(i - len(current_word))  # Start position of word
                    current_word = ""
            else:
                # Regular character, add to current word
                current_word += char

        # Add final word if exists
        if current_word.strip():
            words.append(current_word.strip())
            word_boundaries.append(len(generated_text) - len(current_word))

        if len(words) <= nb_tokens:
            return generated_text

        # Find the position to cut at
        if nb_tokens == 0:
            return ""

        cut_position = word_boundaries[nb_tokens] if nb_tokens < len(word_boundaries) else len(generated_text)
        return generated_text[:cut_position]
