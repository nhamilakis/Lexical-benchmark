#!/usr/bin/env python
"""Configuration classes for metrics computation."""

import typing as t
from dataclasses import dataclass, field


@dataclass
class MetricsConfig:
    """Configuration for metrics computation."""

    # Path configurations
    generation_path: str = "gen/v2"
    reference_path: str = "gen/merged/CHILDES_model.csv"
    frequency_path: str = "metric/gen_freq"
    word_estimation_path: str = "datasets/metric/vocal_month.csv"
    metric_output_path: str = "datasets/metric"

    # Processing parameters
    metrics_list: list[str] = field(
        default_factory=lambda: ["type_token_ratio", "rej_type_rate", "rej_token_rate", "CDI"]
    )
    temperature_list: list[float] = field(default_factory=lambda: [0.3, 0.6, 1.0, 1.5])
    hour_per_year: int = 1000
    chunk_size: int = 3500
    aggregation_months: int = 4
    language: str = "EN"

    # CDI specific parameters
    cdi_threshold: int = 60
    n_bins: int = 12
    sampling_ratio: int = 1

    def get_path_config(self) -> dict[str, str]:
        """Get path configuration dictionary."""
        return {
            "generation": self.generation_path,
            "reference": self.reference_path,
            "frequency": self.frequency_path,
            "word_estimation": self.word_estimation_path,
            "metric": self.metric_output_path,
        }

    def get_non_cdi_metrics(self) -> list[str]:
        """Get list of non-CDI metrics."""
        return [m for m in self.metrics_list if m != "CDI"]

    def is_cdi_enabled(self) -> bool:
        """Check if CDI calculation is enabled."""
        return "CDI" in self.metrics_list


@dataclass
class ThresholdTestConfig:
    """Configuration for CDI threshold testing."""

    # Path configurations
    reference_path: str = "gen/merged/CHILDES_model.csv"
    word_estimation_path: str = "datasets/metric/vocal_month.csv"
    metric_output_path: str = "datasets/metric"

    # Testing parameters
    threshold_list: list[int] = field(default_factory=lambda: [1, 30, 40, 50, 60, 70, 80, 100])
    sampling_ratio: int = 1

    def get_path_config(self) -> dict[str, str]:
        """Get path configuration dictionary."""
        return {
            "reference": self.reference_path,
            "word_estimation": self.word_estimation_path,
            "metric": self.metric_output_path,
        }


@dataclass
class ProcessingFilters:
    """Filtering options for data processing."""

    datasets: list[str] | None = None
    months: list[int] | None = None
    chunks: list[str] | None = None
    models: list[str] | None = None
    temperatures: list[float] | None = None

    def dataset_filter(self, dataset: str) -> bool:
        """Filter function for datasets."""
        return self.datasets is None or dataset in self.datasets

    def month_filter(self, month: int) -> bool:
        """Filter function for months."""
        return self.months is None or month in self.months

    def chunk_filter(self, chunk: str) -> bool:
        """Filter function for chunks."""
        return self.chunks is None or chunk in self.chunks

    def model_filter(self, model: str) -> bool:
        """Filter function for models."""
        return self.models is None or model in self.models

    def temperature_filter(self, temperature: float) -> bool:
        """Filter function for temperatures."""
        return self.temperatures is None or temperature in self.temperatures


@dataclass
class CDIConfig:
    """Configuration specific to CDI calculations."""

    threshold: int
    binned_words: list[list[str]] | None = None
    previous_words: dict[str, float] = field(default_factory=dict)
    word_estimation: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_data_loader(
        cls, data_loader, dataset: str, threshold: int, previous_words: dict[str, float] | None = None
    ) -> "CDIConfig":
        """Create CDI config from data loader results."""
        cdi_result = data_loader.load_cdi_words(dataset)
        word_estimation = data_loader.load_word_estimation_dict(data_loader.path_manager.get_word_estimation_path())

        return cls(
            threshold=threshold,
            binned_words=cdi_result.binned_words,
            previous_words=previous_words or cdi_result.previous_words,
            word_estimation=word_estimation,
        )


@dataclass
class OutputConfig:
    """Configuration for output formatting and saving."""

    output_format: str = "csv"  # csv, parquet, etc.
    include_frequency_data: bool = True
    save_intermediate_results: bool = False
    compression: str | None = None

    def get_output_filename(self, config: MetricsConfig) -> str:
        """Generate output filename based on configuration."""
        return f"metric_{config.hour_per_year}_{config.aggregation_months}_{config.chunk_size}.{self.output_format}"


def create_default_config() -> MetricsConfig:
    """Create default metrics configuration."""
    return MetricsConfig()


def create_threshold_test_config() -> ThresholdTestConfig:
    """Create default threshold testing configuration."""
    return ThresholdTestConfig()


def validate_config(config: MetricsConfig) -> list[str]:
    """Validate configuration and return list of errors."""
    errors = []

    # Validate paths exist (basic check)
    if not config.generation_path:
        errors.append("Generation path cannot be empty")

    if not config.metrics_list:
        errors.append("Metrics list cannot be empty")

    if not config.temperature_list:
        errors.append("Temperature list cannot be empty")

    # Validate parameter ranges
    if config.chunk_size <= 0:
        errors.append("Chunk size must be positive")

    if config.aggregation_months <= 0:
        errors.append("Aggregation months must be positive")

    if config.cdi_threshold <= 0:
        errors.append("CDI threshold must be positive")

    if config.n_bins <= 0:
        errors.append("Number of bins must be positive")

    # Validate temperature values
    for temp in config.temperature_list:
        if temp <= 0:
            errors.append(f"Temperature {temp} must be positive")

    return errors


def load_config_from_dict(config_dict: dict[str, t.Any]) -> MetricsConfig:
    """Load configuration from dictionary."""
    # Filter only known fields
    valid_fields = {f.name for f in MetricsConfig.__dataclass_fields__.values()}
    filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}

    return MetricsConfig(**filtered_dict)
