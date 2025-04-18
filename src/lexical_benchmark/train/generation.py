import string
import typing as t
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from lexical_benchmark import lb_types

from .checkpoint_utils import GenerationCheckpoint
from .trainers.lstm import LSTMConfig, LSTMForLanguageModeling

Model = t.Any


class BatchGenerator:
    """Generator class."""

    def __init__(
        self,
        *,
        model_path: Path,
        tokenizer_name: str,
        device: str,
        use_vllm: bool,
        model_type: lb_types.MODEL_TYPE,
        batch_size: int = 1,
    ) -> None:
        self.device = device
        self.model_path = model_path
        self.use_vllm = use_vllm
        self.model_type = model_type
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size

        if not self.use_vllm:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = self.load_model()

    def load_model(self) -> Model:
        """Load the model."""
        if self.model_type == "lstm":
            config = LSTMConfig.from_pretrained(self.model_path)
            return LSTMForLanguageModeling.from_pretrained(self.model_path, config=config).to(self.device)
        if self.model_type == "gpt2":
            if self.use_vllm:
                return LLM(
                    model=str(self.model_path),
                    tokenizer=self.tokenizer_name,
                    gpu_memory_utilization=0.9,
                    tensor_parallel_size=1,
                )
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            return model.to(self.device)
        return None

    def save_generation(
        self,
        temperature: float,
        gen_attrs: dict,
        *,
        resume: bool = True,
        override: bool = False,
    ) -> None:
        """Generate text from model."""
        # initialize the model
        if resume:
            resume_checkpoint = GenerationCheckpoint.load_intermediate(
                location=self.model_path, temperature=temperature
            )
        if override or not resume_checkpoint:
            resume_checkpoint = GenerationCheckpoint.init_from_args(temperature=temperature, word_counts=gen_attrs)

        if resume_checkpoint.remaining_count() == 0:
            print("No more items require generation, exiting")
            return

        # While items still left to generate
        while resume_checkpoint.remaining_count() > 0:
            next_id, leftover = resume_checkpoint.get_next_gen()
            text, count = self.generate_text(temperature, leftover)
            resume_checkpoint.append_to(gen_id=next_id, text=text, token_count=count)
            print(f"Saving checkpoint of {next_id} to disk")
            resume_checkpoint.save_intermediate(Path("data"))
        print("Completed generation")

    def generate_text(self, temperature: float, nb_tokens: int) -> str:
        """Generate text from model."""
        generated_text = ""
        curr_tokens = 0
        while curr_tokens < nb_tokens:
            new_text = self._generate_sequence(temperature)
            generated_text += new_text
            curr_tokens = self._count_words(generated_text)
            if curr_tokens >= nb_tokens:
                break
        return generated_text, curr_tokens  # type: ignore

    def _generate_sequence(self, temperature) -> str:
        """Generate sequences based on model types."""
        if self.use_vllm:
            return self._generate_vllm(temperature)
        return self._generate_batch_vanilla(temperature)

    def _generate_batch_vanilla(self, temperature: float) -> str:
        """Generate a batch of sentences using GPT2 or LSTM."""
        input_ids = self.tokenizer([""] * self.batch_size, return_tensors="pt", padding=True).input_ids.to(self.device)
        outputs = self.model.generate(
            input_ids,
            max_length=1024,  # reasonable for batch generation
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
        )
        # decode and flatten the batched generation
        generated_text = ""
        for output in outputs:
            generated_text += self.tokenizer.decode(output, skip_special_tokens=True)
        return generated_text  # type: ignore

    def _generate_vllm(self, temperature: float) -> str:
        """Generate next token using vLLM."""
        prompt = ""
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=1024,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        outputs = self.model.generate(
            prompts=prompt,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text

    def _count_words(self, generated_text: str) -> int:
        for char in string.punctuation:
            if char != "|":
                generated_text = generated_text.replace(char, "|")
        words = [word for word in generated_text.split("|") if word.strip()]
        return len(words)

    def _cut_text(self, generated_text: str, nb_tokens: int) -> str:
        text_for_splitting = generated_text
        for char in string.punctuation:
            if char != "|":
                text_for_splitting = text_for_splitting.replace(char, "|")
        tokens = [token for token in text_for_splitting.split("|") if token.strip()]
        return "|".join(tokens[:nb_tokens])
