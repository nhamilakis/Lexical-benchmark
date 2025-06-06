#!/usr/bin/env python
"""Core metric calculation classes for lexical benchmark."""

from dataclasses import dataclass

import pandas as pd

from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.datasets.utils.text_cleaning import char2word, segment_sent
from lexical_benchmark.metrics import normalised_rejection_rates


@dataclass
class CDIResult:
    """Result container for CDI calculations."""

    overall_score: float | None
    binned_scores: list[float] | None
    cumulative_counts: dict[str, float] | None
    frequency_df: pd.DataFrame | None


@dataclass
class MetricResult:
    """Result container for all metric calculations."""

    type_token_ratio: float | None = None
    rej_type_rate: float | None = None
    rej_token_rate: float | None = None
    cdi_result: CDIResult | None = None
    word_count: int = 0


class MetricCalculator:
    """Base metric calculator with common functionality."""

    def __init__(self, word_dict: dataset_utils.DictionairyCleaner, chunk_size: int | None = None) -> None:
        """Initialize calculator with word dictionary and chunk size."""
        self.word_dict = word_dict
        self.chunk_size = chunk_size

    def _prepare_metric_instance(self, data: list[str]) -> normalised_rejection_rates:
        """Prepare metric calculation instance."""
        from lexical_benchmark.metrics.metric import Metric

        return Metric(
            data=segment_sent(data),
            chunk_size=self.chunk_size,
            word_dict=self.word_dict,
        )

    def compute_ttr(self, data: list[str]) -> float:
        """Compute type-token ratio."""
        metric = self._prepare_metric_instance(data)
        return metric.compute_ttr()

    def compute_rejection_rates(self, data: list[str]) -> dict[str, float]:
        """Compute both type and token rejection rates."""
        metric = self._prepare_metric_instance(data)
        return {
            "rej_type_rate": metric.compute_type_rej_rate(),
            "rej_token_rate": metric.compute_token_rej_rate(),
        }

    def compute_all_metrics(self, data: list[str], metrics_list: list[str]) -> dict[str, float]:
        """Compute all requested non-CDI metrics."""
        if not data:
            return {metric: 0.0 for metric in metrics_list}

        # Clean and validate data
        cleaned_data = self._clean_data(data)
        if not cleaned_data:
            return {metric: 0.0 for metric in metrics_list}

        metric = self._prepare_metric_instance(cleaned_data)
        results = {}

        if "type_token_ratio" in metrics_list:
            results["type_token_ratio"] = metric.compute_ttr()
        if "rej_type_rate" in metrics_list:
            results["rej_type_rate"] = metric.compute_type_rej_rate()
        if "rej_token_rate" in metrics_list:
            results["rej_token_rate"] = metric.compute_token_rej_rate()

        return results

    def _clean_data(self, data: list[str]) -> list[str]:
        """Clean and validate input data."""
        cleaned_data = []
        for item in data:
            if isinstance(item, (float, int)):
                if pd.isna(item):
                    continue
                cleaned_data.append(str(item))
            elif isinstance(item, str):
                if item.strip():
                    cleaned_data.append(item)
        return cleaned_data


class CDICalculator:
    """Specialized CDI calculation with state management."""

    def __init__(self, cdi_words: list[str], word_estimation: dict[int, float]) -> None:
        """Initialize CDI calculator with word list and estimation data."""
        self.cdi_words = cdi_words
        self.word_estimation = word_estimation

    def calculate_monthly_score(
        self, texts: list[str], month: int, previous_words: dict[str, float], threshold: int
    ) -> CDIResult:
        """Calculate CDI score for a month with cumulative word tracking."""
        if not self.cdi_words or not texts:
            return CDIResult(None, None, None, None)

        try:
            word_count_est = self.word_estimation.get(month, 0)

            # Calculate overall CDI score
            calculator = self._create_calculator_instance(texts, word_count_est, previous_words)
            cum_counts = calculator.get_combined_counts()
            overall_score = calculator.compute_mean_cdi_score(cum_counts, threshold)
            freq_df = calculator.compute_freq()

            return CDIResult(
                overall_score=overall_score,
                binned_scores=None,  # Will be calculated separately if needed
                cumulative_counts=cum_counts,
                frequency_df=freq_df,
            )

        except Exception as e:
            print(f"Error calculating CDI score: {e}")
            return CDIResult(None, None, None, None)

    def calculate_binned_scores(
        self,
        texts: list[str],
        binned_words: list[list[str]],
        month: int,
        previous_words: dict[str, float],
        threshold: int,
    ) -> list[float]:
        """Calculate CDI scores for each frequency bin."""
        if not binned_words:
            return []

        word_count_est = self.word_estimation.get(month, 0)
        binned_scores = []

        for bin_words in binned_words:
            if not bin_words:
                binned_scores.append(0.0)
                continue

            try:
                # Filter previous words for this bin
                bin_previous = {k: v for k, v in previous_words.items() if k in bin_words}
                calculator = self._create_calculator_instance(texts, word_count_est, bin_previous, bin_words)
                bin_counts = calculator.get_combined_counts()
                bin_score = calculator.compute_mean_cdi_score(bin_counts, threshold)
                binned_scores.append(bin_score)
            except Exception as e:
                print(f"Error calculating binned CDI score: {e}")
                binned_scores.append(0.0)

        return binned_scores

    def _create_calculator_instance(
        self,
        texts: list[str],
        word_count_est: float,
        previous_words: dict[str, float],
        word_list: list[str] | None = None,
    ):
        """Create CDI calculator instance."""
        from lexical_benchmark.metrics.CDI_scores import CDICalculator as CoreCDICalculator

        return CoreCDICalculator(
            CDI_words=word_list or self.cdi_words,
            word_count_est=word_count_est,
            word_list=segment_sent(texts),
            previous_words=previous_words,
        )


class BatchMetricCalculator:
    """Calculate metrics for batches of temperature data."""

    def __init__(self, metric_calculator: MetricCalculator, cdi_calculator: CDICalculator | None = None) -> None:
        """Initialize with metric calculators."""
        self.metric_calculator = metric_calculator
        self.cdi_calculator = cdi_calculator

    def process_temperature_batch(
        self, texts_by_temp: dict[float, list[str]], month: int, metrics_list: list[str], cdi_config: dict | None = None
    ) -> dict[float, MetricResult]:
        """Process all temperatures for a given month/chunk/model combination."""
        results = {}

        for temp, texts in texts_by_temp.items():
            result = MetricResult()

            # Calculate non-CDI metrics
            non_cdi_metrics = [m for m in metrics_list if m != "CDI"]
            if non_cdi_metrics:
                metric_results = self.metric_calculator.compute_all_metrics(texts, non_cdi_metrics)
                result.type_token_ratio = metric_results.get("type_token_ratio")
                result.rej_type_rate = metric_results.get("rej_type_rate")
                result.rej_token_rate = metric_results.get("rej_token_rate")

            # Calculate CDI metrics if enabled
            if "CDI" in metrics_list and self.cdi_calculator and cdi_config:
                result.cdi_result = self.cdi_calculator.calculate_monthly_score(
                    texts=texts,
                    month=month,
                    previous_words=cdi_config["previous_words"],
                    threshold=cdi_config["threshold"],
                )

                # Calculate binned scores if needed
                if cdi_config.get("binned_words"):
                    binned_scores = self.cdi_calculator.calculate_binned_scores(
                        texts=texts,
                        binned_words=cdi_config["binned_words"],
                        month=month,
                        previous_words=cdi_config["previous_words"],
                        threshold=cdi_config["threshold"],
                    )
                    if result.cdi_result:
                        result.cdi_result.binned_scores = binned_scores

            # Calculate word count
            result.word_count = sum(len(char2word(text).split()) for text in texts)
            results[temp] = result

        return results
