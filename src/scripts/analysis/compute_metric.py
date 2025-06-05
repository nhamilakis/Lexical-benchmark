#!/usr/bin/env python
"""Compute core metrics from the generation directory - Refactored Version."""

import argparse
import typing as t

import pandas as pd

from lexical_benchmark import settings
from lexical_benchmark.metrics.calculators import BatchMetricCalculator, CDICalculator, MetricCalculator
from lexical_benchmark.metrics.config import MetricsConfig, OutputConfig, validate_config
from lexical_benchmark.metrics.loaders import DataLoader, PathManager
from lexical_benchmark.metrics.processors import (
    DataProcessor,
    GenerationDataIterator,
    HumanDataProcessor,
    ResultsFormatter,
)
from lexical_benchmark.metrics.state import CDIStateManager


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute lexical benchmark metrics")

    # Path arguments
    parser.add_argument(
        "-g", "--gen_path", type=str, default="archive/gen/v2", help="Relative path to the generated texts"
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        default="archive/gen/merged/CHILDES_model.csv",
        help="Relative path to the human reference data",
    )
    parser.add_argument("--freq_path", type=str, default="metric/gen_freq", help="Relative path to the generated freq")
    parser.add_argument(
        "--word_est_path", type=str, default="datasets/metric/vocal_month.csv", help="Relative path to vocal estimation"
    )
    parser.add_argument("--metric_path", type=str, default="datasets/metric", help="Relative path to save metrics")

    # Metric parameters
    parser.add_argument(
        "--metric_lst",
        type=list,
        default=["type_token_ratio", "rej_type_rate", "rej_token_rate", "CDI"],
        help="Metric list to compute",
    )
    parser.add_argument("--temp_lst", type=list, default=[0.3, 0.6, 1.0, 1.5], help="Temperature list")
    parser.add_argument("-e", "--hour_per_year", default=1000, type=int, help="Estimated yearly exposure hours")
    parser.add_argument("-c", "--chunk_size", default=3500, type=int, help="Chunk size to normalize the scores")
    parser.add_argument(
        "-a",
        "--agg_months",
        type=int,
        default=4,
        help="Number of months to aggregate for rejection rate and TTR computation",
    )
    parser.add_argument("--lang", default="EN", type=str, help="Tested language")

    # CDI parameters
    parser.add_argument("--threshold", default=60, type=int, help="Threshold to compute CDI scores")
    parser.add_argument("-b", "--n_bins", default=12, type=int, help="Frequency bin number for CDI scores")
    parser.add_argument("-s", "--sampling_ratio", type=int, default=1)

    return parser.parse_args()


class MetricsComputer:
    """High-level orchestrator for metrics computation."""

    def __init__(self, config: MetricsConfig) -> None:
        """Initialize with configuration."""
        self.config = config

        # Initialize components
        self.path_manager = PathManager(config.get_path_config())
        self.data_loader = DataLoader(config.sampling_ratio, config.n_bins)
        self.state_manager = CDIStateManager()
        self.data_processor = DataProcessor(config.aggregation_months)
        self.human_processor = HumanDataProcessor(config.aggregation_months)

        # Results storage
        self.all_results = []

    def setup_calculators(self, dataset: str) -> tuple[MetricCalculator, CDICalculator | None]:
        """Setup metric calculators for a dataset."""
        # Load dictionary
        word_dict = self.data_loader.load_dictionary(settings.dataset_name_dict.get(dataset, dataset))

        # Create base metric calculator
        metric_calculator = MetricCalculator(word_dict, self.config.chunk_size)

        # Create CDI calculator if needed
        cdi_calculator = None
        if self.config.is_cdi_enabled():
            word_estimation = self.data_loader.load_word_estimation_dict(self.path_manager.get_word_estimation_path())
            cdi_result = self.data_loader.load_cdi_words(dataset)
            cdi_calculator = CDICalculator(cdi_result.words, word_estimation)

        return metric_calculator, cdi_calculator

    def process_model_generations(self) -> None:
        """Process model-generated data."""
        gen_dir = self.path_manager.base_paths["generation"]

        if not gen_dir.exists():
            print(f"Generation directory {gen_dir} does not exist")
            return

        for dataset_path in gen_dir.iterdir():
            if not dataset_path.is_dir():
                continue

            dataset_name = dataset_path.name
            print(f"Processing dataset: {dataset_name}")

            # Setup calculators for this dataset
            metric_calculator, cdi_calculator = self.setup_calculators(dataset_name)
            batch_calculator = BatchMetricCalculator(metric_calculator, cdi_calculator)

            # Get available months
            base_path = self.path_manager.get_generation_base_path(
                dataset_name, self.config.hour_per_year, self.config.language
            )

            if not base_path.exists():
                print(f"No data found for {dataset_name} at {base_path}")
                continue

            months = [int(month.name) for month in base_path.iterdir() if month.is_dir()]

            # Create data iterator
            data_iterator = GenerationDataIterator(
                self.path_manager,
                dataset_name,
                self.config.hour_per_year,
                self.config.language,
                months,
                self.config.temperature_list,
            )

            # Group data for batch processing
            grouped_data = self.data_processor.group_by_month_chunk_model(data_iterator)
            # Process batches
            batch_results = self.data_processor.process_monthly_batches(
                grouped_data,
                lambda temp_data, month, context: self._process_month_batch(
                    temp_data, month, context, batch_calculator, dataset_name
                ),
            )
            self.all_results.extend(batch_results)
            print(f"Completed processing {dataset_name}")

    def _process_month_batch(
        self,
        temp_data: dict[float, list[str]],
        month: int,
        context: dict,
        batch_calculator: BatchMetricCalculator,
        dataset_name: str,
    ) -> dict[float, t.Any]:
        """Process a single month batch."""
        results = {}

        for temp, texts in temp_data.items():
            try:
                # Prepare CDI configuration if needed
                cdi_config = None
                if self.config.is_cdi_enabled() and batch_calculator.cdi_calculator:
                    previous_words = self.state_manager.get_previous_words(
                        context["dataset"], context["chunk"], context["model"], temp
                    )

                    cdi_result = self.data_loader.load_cdi_words(dataset_name)
                    cdi_config = {
                        "previous_words": previous_words,
                        "threshold": self.config.cdi_threshold,
                        "binned_words": cdi_result.binned_words,
                    }

                # Calculate metrics
                result = batch_calculator.process_temperature_batch(
                    {temp: texts}, month, self.config.metrics_list, cdi_config
                )[temp]

                # Update CDI state if needed
                if cdi_config and result.cdi_result and result.cdi_result.cumulative_counts:
                    self.state_manager.update_words(
                        context["dataset"],
                        context["chunk"],
                        context["model"],
                        temp,
                        result.cdi_result.cumulative_counts,
                    )

                # Save frequency data if available
                if result.cdi_result and result.cdi_result.frequency_df is not None:
                    freq_path = self.path_manager.get_frequency_path(
                        dataset_name,
                        self.config.hour_per_year,
                        self.config.language,
                        month,
                        context["chunk"],
                        context["model"],
                    )
                    freq_path.mkdir(parents=True, exist_ok=True)
                    result.cdi_result.frequency_df.to_csv(freq_path / f"freq_{temp}.csv", index=False)

                results[temp] = result

            except Exception as e:
                print(f"Error processing {context} temp {temp}: {e}")
                continue

        return results

    def process_human_data(self) -> None:
        """Process human reference data."""
        ref_path = self.path_manager.get_reference_path()

        if not ref_path.exists():
            print(f"Reference data not found at {ref_path}")
            return

        ref_data = self.data_loader.load_reference_data(ref_path)

        # Setup calculators for CHILDES
        metric_calculator, cdi_calculator = self.setup_calculators("CHILDES")
        batch_calculator = BatchMetricCalculator(metric_calculator, cdi_calculator)
        # Process human data
        human_results = self.human_processor.process_human_data(
            ref_data, lambda gen, context: self._process_human_month(gen, context, batch_calculator)
        )
        for result in human_results:
            # Extract the MetricResult object from the result dict
            metric_result = result.get("metric_result")
            if metric_result:
                result_dict = {
                    "dataset": result.get("dataset", "CHILDES"),
                    "month": result.get("month"),
                    "chunk": result.get("chunk", "00"),
                    "model": result.get("model_type", "human"),
                    "temp": result.get("temp", "1.0"),
                    "result": metric_result,  # This is the MetricResult object
                }
                self.all_results.append(result_dict)
        print("Completed processing human reference data")

    def _process_human_month(self, gen: pd.DataFrame, context: dict, batch_calculator) -> dict:
        """Process human month data for HumanDataProcessor - returns MetricResult in dict."""
        texts = gen["text"].fillna("").astype(str).tolist()
        temp = 1.0

        if not texts or all(not text.strip() for text in texts):
            return {}

        # Prepare CDI configuration
        cdi_config = None
        if self.config.is_cdi_enabled() and batch_calculator.cdi_calculator:
            previous_words = self.state_manager.get_previous_words("CHILDES", "00", "human", temp)
            cdi_result = self.data_loader.load_cdi_words("CHILDES")
            cdi_config = {
                "previous_words": previous_words,
                "threshold": self.config.cdi_threshold,
                "binned_words": cdi_result.binned_words,
            }

        # Calculate metrics
        metric_result = batch_calculator.process_temperature_batch(
            {temp: texts}, context["month"], self.config.metrics_list, cdi_config
        )[temp]

        # Update CDI state
        if cdi_config and metric_result.cdi_result and metric_result.cdi_result.cumulative_counts:
            self.state_manager.update_words("CHILDES", "00", "human", temp, metric_result.cdi_result.cumulative_counts)

        # Return format compatible with HumanDataProcessor
        return {
            "month": context["month"],
            "metric_result": metric_result,  # Store the MetricResult object here
        }

    def _process_human_month1(self, gen: pd.DataFrame, context: dict, batch_calculator: BatchMetricCalculator) -> dict:
        """Process a single month of human data."""
        texts = gen["text"].fillna("").astype(str).tolist()
        temp = 1.0  # Human data uses temperature 1.0

        # Prepare CDI configuration
        cdi_config = None
        if self.config.is_cdi_enabled() and batch_calculator.cdi_calculator:
            previous_words = self.state_manager.get_previous_words("CHILDES", "00", "human", temp)

            cdi_result = self.data_loader.load_cdi_words("CHILDES")

            cdi_config = {
                "previous_words": previous_words,
                "threshold": self.config.cdi_threshold,
                "binned_words": cdi_result.binned_words,
            }

        # Calculate metrics
        result = batch_calculator.process_temperature_batch(
            {temp: texts}, context["month"], self.config.metrics_list, cdi_config
        )[temp]

        # Update CDI state
        if cdi_config and result.cdi_result and result.cdi_result.cumulative_counts:
            self.state_manager.update_words("CHILDES", "00", "human", temp, result.cdi_result.cumulative_counts)

        # Format result for human data
        formatted_result = {
            "month": context["month"],
            "word_num": gen["sent_len"].sum(),
        }

        if result.type_token_ratio is not None:
            formatted_result["type_token_ratio"] = result.type_token_ratio
        if result.rej_type_rate is not None:
            formatted_result["rej_type_rate"] = result.rej_type_rate
        if result.rej_token_rate is not None:
            formatted_result["rej_token_rate"] = result.rej_token_rate
        if result.cdi_result and result.cdi_result.overall_score is not None:
            formatted_result["CDI"] = result.cdi_result.overall_score
            if result.cdi_result.binned_scores:
                for i, score in enumerate(result.cdi_result.binned_scores):
                    formatted_result[f"CDI_{i}"] = score

        return formatted_result

    def save_results(self) -> None:
        """Save computed metrics to file."""
        if not self.all_results:
            print("No results to save")
            return

        # Format results
        df_results = ResultsFormatter.format_metric_results(
            self.all_results, self.config.metrics_list, self.config.n_bins
        )

        # Determine output path
        if self.config.is_cdi_enabled():
            from lexical_benchmark.datasets.wordstats.data import WordStatsDataset

            cdi_dataset = WordStatsDataset(sampling_ratio=self.config.sampling_ratio)
            output_dir = cdi_dataset.matched_root
        else:
            output_dir = self.path_manager.base_paths["metric"]

        output_config = OutputConfig()
        output_filename = output_config.get_output_filename(self.config)
        output_path = output_dir / output_filename

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_path, index=False)
        print(f"Saved metrics to {output_path}")

    def run(self) -> None:
        """Run the complete metrics computation pipeline."""
        print("Starting metrics computation...")
        print(f"Configuration: {self.config}")

        # Validate configuration
        errors = validate_config(self.config)
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return

        try:
            self.process_model_generations()
            self.process_human_data()
            self.save_results()
            print("Metrics computation completed successfully")

        except Exception as e:
            print(f"Error during metrics computation: {e}")
            raise


def main() -> None:
    """Main entry point for metrics computation."""
    args = parse_args()

    # Convert args to config
    config = MetricsConfig(
        generation_path=args.gen_path,
        reference_path=args.ref_path,
        frequency_path=args.freq_path,
        word_estimation_path=args.word_est_path,
        metric_output_path=args.metric_path,
        metrics_list=args.metric_lst,
        temperature_list=args.temp_lst,
        hour_per_year=args.hour_per_year,
        chunk_size=args.chunk_size,
        aggregation_months=args.agg_months,
        language=args.lang,
        cdi_threshold=args.threshold,
        n_bins=args.n_bins,
        sampling_ratio=args.sampling_ratio,
    )

    computer = MetricsComputer(config)
    computer.run()


if __name__ == "__main__":
    main()
