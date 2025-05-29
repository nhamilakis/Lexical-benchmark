import logging
import typing as t
from pathlib import Path

import clypi.parsers as cp
import numpy as np
import torch
from clypi import Command, Positional, arg

from lexical_benchmark import exc, lb_types, settings
from lexical_benchmark.dataloaders import by_size
from lexical_benchmark.utils import generic as generic_utils

from .array_index_params import GenerationIndex, SlurmIndex
from .generation import BatchGenerator
from .generation_constants import get_token_count_dict

L = None

LogLevelType = t.Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def init_logging(log_level: LogLevelType, log_path: Path, *, log_to_std: bool = False) -> None:
    """Initialise logging."""
    global L  # noqa: PLW0603
    if log_to_std:
        generic_utils.setup_logging(log_level)
    else:
        log_path.parent.mkdir(exist_ok=True, parents=True)
        generic_utils.setup_logging(log_level, log_file=log_path, no_stdout=True)

    L = logging.getLogger(__name__)


def load_generator(
    model_path: Path,
    device: lb_types.DEVICE_TYPE,
    model_type: lb_types.MODEL_TYPE,
    seed: int,
    tokenizer_name: str,
    *,
    use_vllm: bool,
) -> BatchGenerator:
    """Load the generation object."""
    # Setup SEEDs
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return BatchGenerator(
        model_path=model_path,
        tokenizer_name=tokenizer_name,
        device=device,
        use_vllm=use_vllm,
        model_type=model_type,
    )


def convert_bool(arg_string: str) -> bool:
    """Convert the string into bool."""
    return arg_string != "null"


class Single(Command):
    """Run generation in single mode."""

    dataset_name: Positional[t.Literal["stela", "child_realistic"]]
    lang: Positional[str]
    split: Positional[str]
    chunk: Positional[str]
    model_type: Positional[lb_types.MODEL_TYPE]

    # Inherited
    checkpoint_id: str | None = arg(inherited=True, group="generation-params")
    temperature_list: tuple[float] = arg(inherited=True, group="generation-params")
    hour_per_year: int = arg(inherited=True, group="generation-params")
    resume: bool = arg(inherited=True, group="generation-params")
    override: bool = arg(inherited=True, group="generation-params")

    seed: int = arg(inherited=True, group="global-params")
    device: lb_types.DEVICE_TYPE = arg(inherited=True, group="global-params")
    use_vllm: bool = arg(inherited=True, group="global-params")
    save_interval: int = arg(inherited=True, group="global-params")
    debug: bool = arg(inherited=True, group="global-params")
    added_tokens: list[str] = arg(inherited=True, group="global-params")
    model_config_file: Path | None = arg(inherited=True, group="global-params")

    log_to_std: bool = arg(inherited=True, group="logs")
    log_level: LogLevelType = arg(inherited=True, group="logs")

    tokenizer_name: str = arg(inherited=True, group="global-params")

    def make_item(self) -> by_size.BySizeItemsLoader:
        """Make data-item."""
        return by_size.BySizeItemsLoader.load(
            dataset_name=self.dataset_name,
            lang=self.lang,
            split=self.split,
            chunk=self.chunk,
        )

    def model_path(self, item: by_size.BySizeItemsLoader) -> Path:
        """Location of model."""
        model_root = item.model_root / self.model_type
        if self.checkpoint_id:
            model_root = model_root / f"checkpoint-{self.checkpoint_id}"
        return model_root

    def generation_root(self, item: by_size.BySizeItemsLoader) -> Path:
        """Root directory for generation."""
        return item.geneneration_checkpoint_root / self.model_type

    def can_use_vllm(self) -> bool:
        """Check if vllm needs to be used."""
        return bool(self.use_vllm and self.model_type == "gpt2")

    def prep_args(self) -> tuple[by_size.BySizeItemsLoader, BatchGenerator, dict]:
        """Prepare arguments."""
        data_item = self.make_item()
        init_logging(
            log_level=self.log_level,
            log_path=data_item.geneneration_checkpoint_root / "gen.log",
            log_to_std=self.log_to_std,
        )
        L.info("Loading generation parameters...")
        generator = load_generator(
            model_path=self.model_path(data_item),
            device=self.device,
            use_vllm=self.can_use_vllm(),
            model_type=self.model_type,
            tokenizer_name=self.tokenizer_name,
            seed=self.seed,
        )

        token_nb_mapping = get_token_count_dict(
            model_size=int(data_item.split),
            lang=data_item.lang,
            month_estimates=self.hour_per_year,
        )
        return data_item, generator, token_nb_mapping

    async def run(self) -> None:
        """Entrypoint."""
        data_item, generator, token_nb_mapping = self.prep_args()
        L.info("Finish parsing the arguments.")
        for temp in self.temperature_list:
            L.info(f"Generating for temperature={temp}")
            text = generator.save_generation(
                target_dir=data_item.geneneration_checkpoint_root / self.model_type,
                temperature=temp,
                gen_attrs=token_nb_mapping,
                resume=self.resume,
                override=self.override,
            )
            L.info(f"Finished generating text for {temp=}")
            L.debug(f"::{text}")


class ArrayIndex(Command):
    """Run generation from array(to be used with slurm-arrays)."""

    index_file: Positional[Path] = arg(parser=cp.Path(exists=True))
    current_index: Positional[int]

    # Inherited
    seed: int = arg(inherited=True, group="global-params")
    device: lb_types.DEVICE_TYPE = arg(inherited=True, group="global-params")
    use_vllm: bool = arg(inherited=True, group="global-params")
    save_interval: int = arg(inherited=True, group="global-params")
    debug: bool = arg(inherited=True, group="global-params")
    added_tokens: list[str] = arg(inherited=True, group="global-params")
    model_config_file: Path | None = arg(inherited=True, group="global-params")

    log_to_std: bool = arg(inherited=True, group="logs")
    log_level: LogLevelType = arg(inherited=True, group="logs")

    tokenizer_name: str = arg(inherited=True, group="global-params")

    def load_index(self) -> GenerationIndex:
        """Make data-item."""
        slurm_index = SlurmIndex(**self.index_file.read_toml())
        try:
            return slurm_index.index[str(self.current_index)]
        except KeyError as err:
            raise exc.SlurmIndexNotFoundError(index=self.current_index, index_file=self.index_file) from err

    def make_item(self, current_i: GenerationIndex) -> by_size.BySizeItemsLoader:
        """Extract item from index."""
        return by_size.BySizeItemsLoader.load(
            dataset_name=current_i.dataset_name,
            lang=current_i.lang,
            split=current_i.split,
            chunk=current_i.chunk,
        )

    def model_path(self, model_root: Path, current_i: GenerationIndex) -> Path:
        """Location of model."""
        model_root = model_root / current_i.model_type
        checkpoint_id = convert_bool(current_i.checkpoint_id)
        if checkpoint_id:
            model_root = model_root / f"checkpoint-{checkpoint_id}"
        return model_root

    def generation_root(self, item: by_size.BySizeItemsLoader, current_i: GenerationIndex) -> Path:
        """Root directory for generation."""
        return item.geneneration_checkpoint_root / current_i.model_type

    def can_use_vllm(self, model_type: lb_types.MODEL_TYPE) -> bool:
        """Check if vllm needs to be used."""
        return bool(self.use_vllm and model_type == "gpt2")

    def prep_args(self) -> tuple[GenerationIndex, by_size.BySizeItemsLoader, BatchGenerator, dict]:
        """Prepare arguments for generation."""
        current_i = self.load_index()
        data_item = self.make_item(current_i)
        init_logging(
            log_level=self.log_level,
            log_path=data_item.geneneration_checkpoint_root / "gen.log",
            log_to_std=self.log_to_std,
        )
        L.info("Loading generation parameters...")
        generator = load_generator(
            model_path=self.model_path(data_item.model_root, current_i),
            device=self.device,
            use_vllm=self.can_use_vllm(current_i.model_type),
            model_type=current_i.model_type,
            tokenizer_name=self.tokenizer_name,
            seed=self.seed,
        )

        token_nb_mapping = get_token_count_dict(
            model_size=int(data_item.split),
            lang=data_item.lang,
            month_estimates=current_i.hour_per_year,
        )
        return current_i, data_item, generator, token_nb_mapping

    async def run(self) -> None:
        """Entrypoint."""
        current_i, data_item, generator, token_nb_mapping = self.prep_args()
        for temp in current_i.temperature_list:
            text = generator.save_generation(
                target_dir=data_item.geneneration_checkpoint_root / current_i.model_type,
                temperature=temp,
                gen_attrs=token_nb_mapping,
                resume=current_i.resume,
                override=current_i.override,
            )
            L.debug(f"Generated text for {temp=}" + text)


class Generate(Command):
    """Command used to launch generation."""

    subcommand: Single | ArrayIndex

    checkpoint_id: str | None = None
    temperature_list: tuple[float, ...] = arg(
        settings.GENERATION_TEMPERATURES, parser=cp.Tuple(cp.Float(max=1000), num=None)
    )
    hour_per_year: tuple[str, ...] = arg(settings.GENERATION_HPY_ITEMS, parser=cp.Tuple(cp.Str(), num=None))
    seed: int = 562
    use_vllm: bool = False
    save_interval: int = 1024
    resume: bool = True
    override: bool = False
    debug: bool = False  # TODO: debug should generate less text ?
    device: lb_types.DEVICE_TYPE = "cuda"
    added_tokens: list[str] = arg(default_factory=lambda: ["'", "|"], parser=cp.List(cp.Str()))

    model_config_file: Path | None = arg(None, parser=cp.Path(exists=True))
    log_to_std: bool = False
    log_level: LogLevelType = "INFO"

    tokenizer_name: str = "phonemetransformers/GPT2-85M-CHAR-TXT"

    # Introspection
    interactive: bool = arg(default=False, hidden=True)
