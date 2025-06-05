#!/usr/bin/env python
"""Test different CDI thresholds on human reference data - Refactored Version."""

import argparse

import pandas as pd

from lexical_benchmark.metrics.calculators import CDICalculator
from lexical_benchmark.metrics.config import ThresholdTestConfig
from lexical_benchmark.metrics.loaders import DataLoader, PathManager
from lexical_benchmark.metrics.state import CDIStateManager


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test CDI thresholds on human data")

    parser.add_argument(
        "--ref_path",
        type=str,
        default="archive/gen/merged/CHILDES_model.csv",
        help="Relative path to the human reference data",
    )
    parser.add_argument("--metric_path", type=str, default="datasets/metric", help="Relative path to save metrics")
    parser.add_argument(
        "--word_est_path", type=str, default="datasets/metric/vocal_month.csv", help="Relative path to vocal estimation"
    )
    parser.add_argument(
        "--threshold_lst", default=[1, 30, 40, 50, 60, 70, 80, 100], type=list, help="List of thresholds to test"
    )
    parser.add_argument("--sampling_ratio", type=int, default=1, help="Sampling ratio for CDI data")

    return parser.parse_args()


class CDIThresholdTester:
    """Specialized tester for CDI threshold analysis."""

    def __init__(self, config: ThresholdTestConfig) -> None:
        """Initialize with threshold testing configuration."""
        self.config = config

        # Initialize components
        self.path_manager = PathManager(config.get_path_config())
        self.data_loader = DataLoader(config.sampling_ratio)
        self.state_manager = CDIStateManager()

        # Results storage
        self.results = []

    def setup_cdi_calculator(self) -> CDICalculator:
        """Setup CDI calculator for CHILDES dataset."""
        word_estimation = self.data_loader.load_word_estimation_dict(self.path_manager.get_word_estimation_path())
        cdi_result = self.data_loader.load_cdi_words("CHILDES")

        return CDICalculator(cdi_result.words, word_estimation)

    def test_thresholds_on_human_data(self) -> pd.DataFrame:
        """Test different thresholds on human reference data."""
        # Load reference data
        ref_path = self.path_manager.get_reference_path()
        if not ref_path.exists():
            print(f"Reference data not found at {ref_path}")
            return pd.DataFrame()

        ref_data = self.data_loader.load_reference_data(ref_path)

        # Setup CDI calculator
        cdi_calculator = self.setup_cdi_calculator()

        # Process data by month
        results = []
        gen_grouped = ref_data.groupby("month")

        for month, gen in gen_grouped:
            print(f"Processing month {month}")

            # Get previous words state
            previous_words = self.state_manager.get_previous_words("CHILDES", "00", "human", 1.0)

            # Prepare text data
            texts = gen["text"].fillna("").astype(str).tolist()
            word_count = gen["sent_len"].sum()

            # Test all thresholds for this month
            month_result = {"month": month, "word_count": word_count}

            # try:
            # Calculate CDI result once
            cdi_result = cdi_calculator.calculate_monthly_score(
                texts=texts,
                month=month,
                previous_words=previous_words,
                threshold=self.config.threshold_list[0],  # Use first threshold for cumulative counts
            )

            if cdi_result.cumulative_counts is None:
                print(f"Warning: No CDI counts for month {month}")
                continue

            # Test each threshold using the same cumulative counts
            for threshold in self.config.threshold_list:
                score = self._compute_threshold_score(cdi_result.cumulative_counts, threshold)
                month_result[f"threshold_{threshold}"] = score

            # Update state with cumulative counts
            self.state_manager.update_words("CHILDES", "00", "human", 1.0, cdi_result.cumulative_counts)

            results.append(month_result)

            """
            except Exception as e:
                print(f"Error processing month {month}: {e}")
                continue
            """
        if not results:
            return pd.DataFrame()

        # Convert to DataFrame
        df_results = pd.DataFrame(results)

        # Add metadata columns
        df_results = df_results.assign(dataset="CHILDES", chunk="00", model_type="human", temp="1.0")

        # Reorder columns
        threshold_cols = [f"threshold_{t}" for t in self.config.threshold_list]
        col_order = ["month", "word_count"] + threshold_cols + ["dataset", "chunk", "model_type", "temp"]
        df_results = df_results[col_order]

        return df_results

    def _compute_threshold_score(self, cumulative_counts: dict[str, float], threshold: int) -> float:
        """Compute CDI score for a specific threshold."""
        if not cumulative_counts:
            return 0.0

        scores = [1 if count >= threshold else 0 for count in cumulative_counts.values()]
        return sum(scores) / len(scores) if scores else 0.0

    def save_results(self, results: pd.DataFrame) -> None:
        """Save threshold test results."""
        if results.empty:
            print("No results to save")
            return

        output_path = self.path_manager.base_paths["metric"] / "metric_CDI_threshold_test.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results.to_csv(output_path, index=False)
        print(f"Saved threshold test results to {output_path}")

    def run(self) -> None:
        """Run the complete threshold testing pipeline."""
        print("Starting CDI threshold testing...")
        print(f"Testing thresholds: {self.config.threshold_list}")

        try:
            results = self.test_thresholds_on_human_data()
            self.save_results(results)

            if not results.empty:
                print(f"Successfully tested {len(self.config.threshold_list)} thresholds")
                print(f"Processed {len(results)} months of data")

            else:
                print("No results generated")

        except Exception as e:
            print(f"Error during threshold testing: {e}")
            raise


def main() -> None:
    """Main entry point for threshold testing."""
    args = parse_args()

    # Convert args to config
    config = ThresholdTestConfig(
        reference_path=args.ref_path,
        word_estimation_path=args.word_est_path,
        metric_output_path=args.metric_path,
        threshold_list=args.threshold_lst,
        sampling_ratio=args.sampling_ratio,
    )

    tester = CDIThresholdTester(config)
    tester.run()


if __name__ == "__main__":
    main()
