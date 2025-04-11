import string
import typing as t
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from lexical_benchmark import lb_types
from lexical_benchmark.train.generators.lstm import LSTMConfig, LSTMForLanguageModeling

Model = t.Any

"""
0. resume and override logic
2. parallel decoding
"""


class BatchGenerator:
    """Generator class."""

    def __init__(
        self,
        model_path: Path,
        tokenizer_name: str,
        device: str,
        use_vllm: bool,
        model_type: lb_types.MODEL_TYPE,
        temp: float,
        nb_tokens: int,
    ) -> None:
        self.device = device
        self.model_path = model_path
        self.use_vllm = use_vllm
        self.model_type = model_type
        self.temp = temp
        self.nb_tokens = nb_tokens
        self.tokenizer_name = tokenizer_name
        if not self.use_vllm:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = self.load_model()

    def load_model(self) -> Model:
        """Load the model."""
        if self.model_type == "lstm":
            config = LSTMConfig.from_pretrained(self.model_path)
            model = LSTMForLanguageModeling.from_pretrained(self.model_path, config=config)
            return model.to(self.device)
        if self.model_type == "gpt2":
            if self.use_vllm:
                self.model = LLM(
                    model=str(self.model_path),
                    tokenizer=self.tokenizer_name,
                    gpu_memory_utilization=0.9,
                    tensor_parallel_size=1,
                )
                return self.model
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            return model.to(self.device)
        return None

    def generate_text(self) -> str:
        """Generate text from model."""
        # Initialize or load existing generated text
        generated_text = ""
        curr_tokens = 0

        # Generate text sentence by sentence until we reach the desired token count
        while curr_tokens < self.nb_tokens:
            # Generate new text
            new_text = self._generate_vllm() if self.use_vllm else self._generate_vanilla()
            generated_text += new_text
            curr_tokens = self._count_words(generated_text)
        if curr_tokens > self.nb_tokens:
            generated_text = self._cut_text(generated_text)
        return generated_text

    def _generate_vanilla(self) -> str:
        """Generate next token using vanilla generation."""
        input_ids = self.tokenizer("", return_tensors="pt").input_ids.to(self.device)
        output = self.model.generate(
            input_ids,
            max_length=1024,  # Maximum length limit
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=self.temp,
        )
        # Decode the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    def _generate_vllm(self) -> str:
        """Generate next token using vLLM."""
        prompt = ""
        sampling_params = SamplingParams(
            temperature=self.temp,
            max_tokens=1024,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        outputs = self.model.generate(
            prompts=prompt,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        generated_text = outputs[0].outputs[0].text
        return generated_text

    def _count_words(self, generated_text: str) -> int:
        # Replace all punctuation with pipe character
        for char in string.punctuation:
            if char != "|":  # Avoid replacing existing pipe characters
                generated_text = generated_text.replace(char, "|")
        # Split by pipe character and count non-empty elements
        words = [word for word in generated_text.split("|") if word.strip()]
        return len(words)

    def _cut_text(self, generated_text: str) -> str:
        """Cut the generated text to the desired number of tokens."""
        # Apply the same replacements as in count_words to get consistent tokenization
        text_for_splitting = generated_text
        for char in string.punctuation:
            if char != "|":  # Avoid replacing existing pipe characters
                text_for_splitting = text_for_splitting.replace(char, "|")
        # Split into tokens
        tokens = [token for token in text_for_splitting.split("|") if token.strip()]
        truncated_tokens = tokens[: self.nb_tokens]
        # Join the tokens back with spaces
        return "|".join(truncated_tokens)

    def save_text(self, nb_tokens: int, target_file: Path, resume: bool = True, override: bool = False) -> None:
        """Generate data from model."""
        return generated_text
