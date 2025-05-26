import logging
import string
import typing as t
import warnings
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import vllm  # type: ignore[missing-dependency]
    from vllm import LLM, SamplingParams  # type: ignore[missing-dependency]
except ImportError:
    warnings.warn("'vllm' could not be imported, is it not installed ?", category=ImportWarning, stacklevel=1)
    vllm = None
    LLM, SamplingParams = (None, None)

from lexical_benchmark import dataloaders, lb_types
from lexical_benchmark.text_lib import txt_utils

from .checkpoint_utils import ESTIMATION_MONTH_KEY_TYPE, GenerationCheckpoint, GenerationsStruct
from .trainers.lstm import LSTMConfig, LSTMForLanguageModeling

Model = t.Any

L = logging.getLogger(__name__)


class BatchGenerator:
    """Generator class."""

    def __init__(
        self,
        *,
        model_path: Path,
        tokenizer_name: str,
        device: lb_types.DEVICE_TYPE,
        use_vllm: bool,
        model_type: lb_types.MODEL_TYPE,
        batch_size: int = 1,
    ) -> None:
        self.device = device

        self.use_vllm = use_vllm
        self.model_type = model_type
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size

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
        # initialize the model
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

        # While items still left to generate
        L.info("Generating rest of the data...")
        while resume_checkpoint.remaining_count() > 0:
            next_id, leftover = resume_checkpoint.get_next_gen()
            resume_checkpoint = self.generate_checkpoint_text(
                temperature=temperature,
                nb_tokens=leftover,
                index=next_id,
                checkpoint=resume_checkpoint,
            )

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
        print(generated_text)
        return generated_text, curr_tokens

    def generate_checkpoint_text(
        self, temperature: float, nb_tokens: int, index: ESTIMATION_MONTH_KEY_TYPE, checkpoint: GenerationCheckpoint
    ) -> GenerationCheckpoint:
        """Generate text from model, and save it into the checkpoint."""
        curr_tokens = 0
        while curr_tokens < nb_tokens:
            new_text = self._generate_sequence(temperature)
            count = self._count_words(new_text)
            checkpoint.append_text_list(index, new_text, count)
            curr_tokens += count
            if curr_tokens >= nb_tokens:
                break
        return checkpoint

    def _generate_sequence(self, temperature) -> str:
        """Generate sequences based on model types."""
        if self.use_vllm and vllm:
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
        return generated_text

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


def build_text_dataset(datasets: tuple[str, ...] = ("stela",), langs=("EN",)) -> None:
    """Build text dataset from generation checkpoints."""
    items_iter: t.Iterable[dataloaders.generation_loaders.GenerationCheckpointLoader] = (
        dataloaders.generation_loaders.GenerationCheckpointLoader.iter_items(
            datasets=datasets,
            langs=langs,
        )
    )
    for item in items_iter:
        if item.is_finished():
            checkpoint: GenerationCheckpoint = item.load_final()
            for (estim, month), struct in checkpoint.gen_items.items():
                text_item: dataloaders.generation_loaders.GenerationItemsLoader = (
                    dataloaders.generation_loaders.GenerationItemsLoader.load(
                        dataset_name=item.dt_cfg.dataset_name,
                        lang=item.lang,
                        model_type=item.model_type,
                        estimation_type=estim,
                        month=month,
                        temperature=item.temperature,
                    )
                )
                trimmed_text = txt_utils.trim_sentence_list(struct["text"], struct["target_count"])
                text_item.text_file.safe_append_text("\n".join(trimmed_text))
