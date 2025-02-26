#!/usr/bin/env python
"""Compute core metrics from the generation directory."""
import argparse
from pathlib import Path

import pandas as pd

from lexical_benchmark import settings
from lexical_benchmark.datasets.utils.text_cleaning import segment_sent
from lexical_benchmark.datasets.wordstats.data import WordStatsDataset
from lexical_benchmark.stats.CDI_scores import CDICalculator
from lexical_benchmark.stats.metric import WordDictManager, load_dict


def parse_args():
    # Run parameters
    parser = argparse.ArgumentParser(description="compute metrics")

    parser.add_argument(
        "--ref_path",
        type=str,
        default="gen/merged/CHILDES_model.csv",
        help="relative path to the human reference data",
    )
    parser.add_argument(
        "--metric_path",
        type=str,
        default="datasets/metric",
        help="relative path to save metrics",
    )
    parser.add_argument(
        "--word_est_path",
        type=str,
        default="datasets/metric/vocal_month.csv",
        help="relative path to vocal estimation"
    )
    parser.add_argument(
        "--threshold_lst",
        default=[1,30,40,50,60,70,80,100],
        type=list,
        help="threshold to compute CDI scores"
        )
    return parser.parse_args()



class MetricsProcessor:
    """Compute and manage metrics for both model-generated and human data."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the MetricsProcessor with command line arguments."""
        self.args = args
        self.paths = self._setup_paths()
        self.word_est_dict = self._load_word_est_dict()
        self.CDI_enabled = True
        self.CDI_month_dict: dict[str, dict] = {"dataset": {"chunk": {"model_type": {"temp": {}}}}}
        self.score_all = pd.DataFrame()
        self.threshold_lst = args.threshold_lst

    def _setup_paths(self) -> dict[str, Path]:
        """Set up all necessary paths."""
        return {
            "metric_dir": settings.PATH.DATA_DIR / self.args.metric_path,
            "ref_dir": settings.PATH.DATA_DIR / self.args.ref_path,
            "word_est_dir": settings.PATH.DATA_DIR / self.args.word_est_path,
        }

    def _load_word_est_dict(self) -> dict[int, float]:
        """Load and return the word estimation dictionary."""
        df_est = pd.read_csv(self.paths["word_est_dir"])
        word_est_dict = dict(zip(df_est["month"], df_est["child_month_est"]))
        print(f"Monthly production estimation loaded {word_est_dict}")
        return word_est_dict

    def load_CDI_words(self, dataset: str) -> tuple[list[str], dict[str, int]]:
        """Load CDI words for different datasets."""
        if self.CDI_enabled:
            CDIdataset = WordStatsDataset()
            # load based on differnet dataset dict
            if dataset == "STELATranscriptions2":
                data = CDIdataset.matched_frequencies_exp.machine.read_csv()
            if dataset == "CHILDES":
                data = CDIdataset.matched_frequencies_exp.cdi.read_csv()
            if dataset == "ChildRealistic":
                data = CDIdataset.matched_frequencies_exp.human_realistc.read_csv()
            CDI_words = data['word'].to_list()
            return CDI_words, dict.fromkeys(CDI_words, 0)
        return [], {}

    def load_word_count_est(self, month: int) -> float:
        """Load word count estimation for each month."""
        return self.word_est_dict.get(month, 0) if self.CDI_enabled else 0


    def _compute_monthly_CDI(
        self,
        sent_lst: list[str],
        CDI_words: list[str],
        word_count_est: float,
        previous_words: dict[str, int],
        threshold_lst: list[int]
    ) -> tuple[list[float] | None, dict | None]:
        """Compute monthly CDI scores if enabled."""
        if not self.CDI_enabled:
            return None, None

        calculator = CDICalculator(
            CDI_words=CDI_words,
            word_count_est=word_count_est,
            word_list=segment_sent(sent_lst),
            previous_words=previous_words,
        )

        try:
            cum_counts = calculator.get_combined_counts()
            df = calculator.compute_freq()
            cdi_scores = []
            for threshold in threshold_lst:
                cdi_score = calculator.compute_mean_cdi_score(cum_counts, threshold)
                cdi_scores.append(cdi_score)
            return cdi_scores, cum_counts,df
        except Exception as e:
            print(f"Error calculating CDI scores: {e}")
            return None, None, None

    def _compute_human_metrics(self, ref_data: pd.DataFrame, CDI_words: list[str], word_dict: dict) -> pd.DataFrame:
        """Compute metrics for human reference data with thresholds as columns."""
        scores = []
        gen_grouped = ref_data.groupby("month")

        # Process each month
        for month, gen in gen_grouped:
            word_dict_manager = WordDictManager(
                dataset="CHILDES",
                chunk="00",
                model_type="human",
                temp="1.0"
            )

            word_count_est = self.load_word_count_est(month)
            previous_words = word_dict_manager.load_word_dict(self.CDI_month_dict)
            sent_lst = gen["text"].fillna("").astype(str).tolist()

            # Get CDI scores for all thresholds
            cdi_scores, cum_counts,df = self._compute_monthly_CDI(
                sent_lst, 
                CDI_words, 
                word_count_est,
                previous_words,
                self.threshold_lst
            )

            if cdi_scores is not None:
                # Create a row with month and word count
                row = {
                    "month": month,
                    "word_count": gen["sent_len"].sum()
                }
                # Add threshold scores as separate columns
                for threshold, score in zip(self.threshold_lst, cdi_scores):
                    row[f"threshold_{threshold}"] = score
                scores.append(row)

            if cum_counts is not None:
                self.CDI_month_dict = word_dict_manager.write_word_dict(
                    cum_counts,
                    self.CDI_month_dict
                )

        if not scores:
            return pd.DataFrame()

        # Create DataFrame from list of dictionaries
        score = pd.DataFrame(scores)
        # Add constant columns
        score = score.assign(
            dataset="CHILDES",
            chunk="00",
            model_type="human",
            temp="1.0"
        )
        # Reorder columns
        threshold_cols = [f"threshold_{t}" for t in self.threshold_lst]
        col_order = ["month", "word_count"] + threshold_cols + ["dataset", "chunk", "model_type", "temp"]
        score = score[col_order]
        return score



    def _process_month(
        self, month: Path, dataset: Path, CDI_words: list[str], word_dict: dict, word_count_est: float
    ) -> None:
        """Process data for a specific month."""
        for chunk in month.iterdir():
            for model in chunk.iterdir():
                gen_file = model / "gen.csv"
                if not gen_file.exists():
                    continue

                gen = pd.read_csv(gen_file)
                info_dict = {
                    "dataset": settings.dataset_name_dict[dataset.name],
                    "month": month.name,
                    "chunk": chunk.name,
                    "model_type": model.name,
                }

                score, self.CDI_month_dict = self._compute_model_metrics(
                    gen=gen,
                    info_dict=info_dict,
                    CDI_words=CDI_words,
                    word_dict=word_dict,
                    word_count_est=word_count_est,
                )

                self.score_all = pd.concat([self.score_all, score])
                print(f"Finish computing metrics from {model.relative_to(self.paths['gen_dir'])}")

    def process_human_data(self) -> None:
        """Process human reference data and compute metrics."""
        ref_data = pd.read_csv(self.paths["ref_dir"])
        CDI_words, previous_words = self.load_CDI_words("CHILDES")
        word_dict = load_dict("CHILDES")

        self.score_all = self._compute_human_metrics(ref_data=ref_data, CDI_words=CDI_words, word_dict=word_dict)

    def save_results(self) -> None:
        """Save computed metrics to file."""
        if self.score_all.empty:
            print("Warning: No results to save")
            return

        output_path = self.paths["metric_dir"] / f"metric_CDI_threshold_test.csv"
        self.score_all.to_csv(output_path, index=False)
        print(f"Saved metrics to {output_path}")

    def run(self) -> None:
        """Run the complete metrics computation pipeline."""
        self.process_human_data()
        self.save_results()


def main() -> None:
    """Main entry point for metrics computation."""
    args = parse_args()
    # loop over different thresholds
    computer = MetricsProcessor(args)
    computer.run()

if __name__ == "__main__":
    main()
