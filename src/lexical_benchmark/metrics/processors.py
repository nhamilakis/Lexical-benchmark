#!/usr/bin/env python
"""Data processing and aggregation utilities."""

import typing as t
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from lexical_benchmark.datasets.utils.text_cleaning import char2word


@dataclass
class GenerationData:
    """Container for generation data item."""

    dataset: str
    month: int
    chunk: str
    model: str
    temperature: float
    texts: list[str]


@dataclass
class ProcessingContext:
    """Context information for error handling."""

    dataset: str
    month: int
    chunk: str
    model: str
    temperature: float | None = None


@dataclass
class ProcessingResult:
    """Result container with error handling."""

    success: bool
    data: t.Any | None = None
    error: str | None = None
    context: ProcessingContext | None = None


class GenerationDataIterator:
    """Iterator for generation data to manage memory efficiently."""

    def __init__(
        self,
        path_manager,
        dataset: str,
        hour_per_year: int,
        lang: str,
        months: list[int],
        temps: list[float],
        chunk_filter: t.Callable[[str], bool] | None = None,
        model_filter: t.Callable[[str], bool] | None = None,
    ) -> None:
        """Initialize iterator with filtering options."""
        self.path_manager = path_manager
        self.dataset = dataset
        self.hour_per_year = hour_per_year
        self.lang = lang
        self.months = months
        self.temps = temps
        self.chunk_filter = chunk_filter or (lambda x: True)
        self.model_filter = model_filter or (lambda x: True)

    def __iter__(self) -> t.Iterator[GenerationData]:
        """Iterate through all generation data combinations."""
        base_path = self.path_manager.get_generation_base_path(self.dataset, self.hour_per_year, self.lang)

        if not base_path.exists():
            return

        for month in self.months:
            month_path = base_path / str(month)
            if not month_path.exists():
                continue

            for chunk_path in month_path.iterdir():
                if not chunk_path.is_dir() or not self.chunk_filter(chunk_path.name):
                    continue

                for model_path in chunk_path.iterdir():
                    if not model_path.is_dir() or not self.model_filter(model_path.name):
                        continue

                    gen_file = model_path / "gen.csv"
                    if not gen_file.exists():
                        continue

                    try:
                        gen_df = pd.read_csv(gen_file)

                        for temp in self.temps:
                            temp_col = f"unprompted_{temp}"
                            if temp_col not in gen_df.columns:
                                continue

                            texts = gen_df[temp_col].apply(char2word).tolist()

                            yield GenerationData(
                                dataset=self.dataset,
                                month=month,
                                chunk=chunk_path.name,
                                model=model_path.name,
                                temperature=temp,
                                texts=texts,
                            )

                    except Exception as e:
                        print(f"Error loading {gen_file}: {e}")
                        continue


class DataProcessor:
    """Handle data aggregation and processing logic."""

    def __init__(self, aggregation_months: int = 1) -> None:
        """Initialize processor with aggregation settings."""
        self.aggregation_months = aggregation_months

    def aggregate_texts_by_group(
        self, data_iterator: t.Iterator[GenerationData]
    ) -> dict[tuple, dict[float, list[str]]]:
        """Aggregate text data across months for each unique group combination."""
        if self.aggregation_months <= 1:
            return {}

        grouped_data: dict[tuple, dict[float, list[str]]] = defaultdict(lambda: defaultdict(list))

        for data_item in data_iterator:
            agg_group = (data_item.month - 1) // self.aggregation_months
            group_key = (data_item.dataset, data_item.chunk, data_item.model, agg_group)
            grouped_data[group_key][data_item.temperature].extend(data_item.texts)

        return grouped_data

    def group_by_month_chunk_model(
        self, data_iterator: t.Iterator[GenerationData]
    ) -> dict[tuple, dict[float, list[str]]]:
        """Group data by month/chunk/model for batch processing."""
        grouped_data: dict[tuple, dict[float, list[str]]] = defaultdict(lambda: defaultdict(list))

        for data_item in data_iterator:
            group_key = (data_item.dataset, data_item.month, data_item.chunk, data_item.model)
            grouped_data[group_key][data_item.temperature] = data_item.texts

        return grouped_data

    def process_aggregated_batches(
        self,
        aggregated_data: dict[tuple, dict[float, list[str]]],
        batch_processor: t.Callable[[dict[float, list[str]], dict], dict[float, t.Any]],
    ) -> list[dict]:
        """Process aggregated data batches."""
        results = []

        for group_key, temp_data in aggregated_data.items():
            dataset, chunk, model, agg_group = group_key

            try:
                batch_results = batch_processor(
                    temp_data,
                    {
                        "dataset": dataset,
                        "chunk": chunk,
                        "model": model,
                        "agg_group": agg_group,
                    },
                )

                for temp, result in batch_results.items():
                    results.append(
                        {
                            "dataset": dataset,
                            "chunk": chunk,
                            "model": model,
                            "agg_group": agg_group,
                            "temp": temp,
                            "result": result,
                        }
                    )

            except Exception as e:
                print(f"Error processing batch {group_key}: {e}")
                continue

        return results

    def process_monthly_batches(
        self,
        grouped_data: dict[tuple, dict[float, list[str]]],
        batch_processor: t.Callable[[dict[float, list[str]], int, dict], dict[float, t.Any]],
    ) -> list[dict]:
        """Process monthly data batches."""
        results = []

        for group_key, temp_data in grouped_data.items():
            dataset, month, chunk, model = group_key

            try:
                batch_results = batch_processor(
                    temp_data,
                    month,
                    {
                        "dataset": dataset,
                        "month": month,
                        "chunk": chunk,
                        "model": model,
                    },
                )

                for temp, result in batch_results.items():
                    results.append(
                        {
                            "dataset": dataset,
                            "month": month,
                            "chunk": chunk,
                            "model": model,
                            "temp": temp,
                            "result": result,
                        }
                    )

            except Exception as e:
                print(f"Error processing batch {group_key}: {e}")
                continue

        return results


class HumanDataProcessor:
    """Specialized processor for human reference data."""

    def __init__(self, aggregation_months: int = 1) -> None:
        """Initialize with aggregation settings."""
        self.aggregation_months = aggregation_months

    def process_human_data(
        self, ref_data: pd.DataFrame, processor_func: t.Callable[[pd.DataFrame], dict]
    ) -> list[dict]:
        """Process human reference data by month groups."""
        results = []
        gen_grouped = ref_data.groupby("month")

        # Pre-compute aggregated data if needed
        agg_data = {}
        if self.aggregation_months > 1:
            agg_data = self._compute_aggregated_human_data(gen_grouped)

        for month, gen in gen_grouped:
            try:
                # Determine if we should use aggregated data
                agg_group = (month - 1) // self.aggregation_months if self.aggregation_months > 1 else None

                result = processor_func(
                    gen,
                    {
                        "month": month,
                        "agg_group": agg_group,
                        "agg_data": agg_data.get(agg_group, {}) if agg_data else {},
                    },
                )

                result.update({"dataset": "CHILDES", "month": month, "chunk": "00", "model": "human", "temp": "1.0"})

                results.append(result)

            except Exception as e:
                print(f"Error processing human data for month {month}: {e}")
                continue

        return results

    def _compute_aggregated_human_data(self, gen_grouped) -> dict[int, dict]:
        """Compute aggregated metrics for human data."""
        agg_data = {}

        for month in gen_grouped.groups:
            agg_group = (month - 1) // self.aggregation_months

            if agg_group not in agg_data:
                group_months = range(
                    agg_group * self.aggregation_months + 1, (agg_group + 1) * self.aggregation_months + 1
                )

                # Collect all text data for this group
                group_texts = []
                total_word_count = 0

                for m in group_months:
                    if m in gen_grouped.groups:
                        month_data = gen_grouped.get_group(m)
                        texts = month_data["text"].fillna("").astype(str).tolist()
                        group_texts.extend(texts)
                        total_word_count += month_data["sent_len"].sum()

                agg_data[agg_group] = {"texts": group_texts, "word_count": total_word_count}

        return agg_data


class ResultsFormatter:
    """Format processing results into final output structure."""

    @staticmethod
    def format_metric_results(results: list[dict], metrics_list: list[str], n_bins: int = 12) -> pd.DataFrame:
        """Format metric results into DataFrame."""
        if not results:
            return pd.DataFrame()

        formatted_rows = []

        # Create column names
        metric_columns = metrics_list.copy()
        if "CDI" in metric_columns:
            metric_columns.extend([f"CDI_{i}" for i in range(n_bins)])

        for result_dict in results:
            result = result_dict.get("result")
            if not result:
                continue

            row = {
                "dataset": result_dict.get("dataset"),
                "month": result_dict.get("month"),
                "chunk": result_dict.get("chunk"),
                "model_type": result_dict.get("model"),
                "temp": result_dict.get("temp"),
            }

            # Add metric values
            if hasattr(result, "type_token_ratio"):
                row["type_token_ratio"] = result.type_token_ratio
            if hasattr(result, "rej_type_rate"):
                row["rej_type_rate"] = result.rej_type_rate
            if hasattr(result, "rej_token_rate"):
                row["rej_token_rate"] = result.rej_token_rate
            if hasattr(result, "cdi_result") and result.cdi_result:
                row["CDI"] = result.cdi_result.overall_score
                if result.cdi_result.binned_scores:
                    for i, score in enumerate(result.cdi_result.binned_scores):
                        row[f"CDI_{i}"] = score
            if hasattr(result, "word_count"):
                row["word_num"] = result.word_count

            formatted_rows.append(row)

        return pd.DataFrame(formatted_rows)
