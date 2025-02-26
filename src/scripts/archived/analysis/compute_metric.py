#!/usr/bin/env python
"""Compute core metrics from the generation directory."""

import argparse
import typing as t
from collections import defaultdict
from pathlib import Path

import pandas as pd

from lexical_benchmark import settings
from lexical_benchmark.datasets.utils.text_cleaning import char2word, segment_sent
from lexical_benchmark.datasets.wordstats.data import WordStatsDataset
from lexical_benchmark.stats.CDI_scores import CDICalculator
from lexical_benchmark.stats.metric import Metric, WordDictManager, load_dict, remap_bins


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="compute metrics")
    parser.add_argument(
        "-g",
        "--gen_path",
        type=str,
        default="gen/v2",
        help="relative path to the generated texts",
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        default="gen/merged/CHILDES_model.csv",
        help="relative path to the human reference data",
    )
    parser.add_argument(
        "--freq_path",
        type=str,
        default="metric/gen_freq",
        help="relative path to the generated freq",
    )
    parser.add_argument(
        "--word_est_path",
        type=str,
        default="datasets/metric/vocal_month.csv",
        help="relative path to vocal estimation",
    )
    parser.add_argument(
        "--metric_path",
        type=str,
        default="datasets/metric",
        help="relative path to save metrics",
    )
    parser.add_argument(
        "--metric_lst",
        type=list,
        default=["type_token_ratio", "rej_type_rate", "rej_token_rate", "CDI"],
        help="metric list; ttr,rej_type_rate,CDI",
    )
    parser.add_argument("--temp_lst", type=list, default=[0.3, 0.6, 1.0, 1.5], help="temperature list")
    parser.add_argument("-e", "--hour_per_year", default=1000, type=int, help="Estimated yearly exposure hours")
    parser.add_argument("-c", "--chunk_size", default=3500, type=int, help="Chunk size to normalize the scores")
    parser.add_argument("--threshold", default=60, type=int, help="threshold to compute CDI scores")
    parser.add_argument("-b", "--n_bins", default=12, type=int, help="freq bin number for CDI scores")
    parser.add_argument("-s", "--sampling_ratio", type=int, default=1)
    parser.add_argument("--lang", default="EN", type=str, help="tested language")
    parser.add_argument(
        "-a",
        "--agg_months",
        type=int,
        default=4,
        help="Number of months to aggregate for rejection rate and TTR computation",
    )
    return parser.parse_args()



class MetricsProcessor:
    """Compute and manage metrics for both model-generated and human data."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the MetricsProcessor with command line arguments."""
        self.args = args
        self.paths = self._setup_paths()
        self.word_est_dict = self._load_word_est_dict()
        self.CDI_enabled = "CDI" in args.metric_lst
        self.CDI_month_dict: dict[str, dict] = {"dataset": {"chunk": {"model_type": {"temp": {}}}}}
        self.score_all = pd.DataFrame()
        self.threshold = args.threshold
        self.non_cdi_metrics = ["type_token_ratio", "rej_type_rate", "rej_token_rate"]
        self.n_bins = args.n_bins
        self.sampling_ratio = args.sampling_ratio

    def _setup_paths(self) -> dict[str, Path]:
        """Set up all necessary paths."""
        return {
            "gen_dir": settings.PATH.DATA_DIR / self.args.gen_path,
            "freq_dir": settings.PATH.DATA_DIR / self.args.freq_path,
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

    def load_CDI_words(self,dataset: str) -> tuple[list[str], list[list[str]], dict[str, int]]:
        """Load CDI words for different datasets."""
        if self.CDI_enabled:
            CDIdataset = WordStatsDataset(sampling_ratio=self.sampling_ratio)
            # Load based on different dataset dict
            if dataset == "STELATranscriptions2":
                data = CDIdataset.matched_frequencies_exp.machine.read_csv()
            elif dataset == "CHILDES":
                data = CDIdataset.matched_frequencies_exp.cdi.read_csv()
            elif dataset == "ChildRealistic":
                data = CDIdataset.matched_frequencies_exp.human_realistc.read_csv()

            # Get full list of CDI words
            CDI_words = data["word"].to_list()
            # Get binned word lists
            CDI_words_lst = remap_bins(data, self.n_bins)
            print("Finished mapping to different bins")
            # Initialize previous words dict with all words
            previous_words = dict.fromkeys(CDI_words, 0)

            return CDI_words, CDI_words_lst, previous_words
        return [], [], {}

    def load_word_count_est(self, month: int) -> float:
        """Load word count estimation for each month."""
        return self.word_est_dict.get(month, 0) if self.CDI_enabled else 0

    def _aggregate_by_group(self, dataset_path: Path, word_dict: dict[str, t.Any]) -> dict[tuple, list[str]]:
        """Aggregate text data across months for each unique group combination."""
        grouped_data: dict[tuple, list[str]] = defaultdict(list)
        base_path = dataset_path / f"{self.args.hour_per_year}_hour_per_year" / self.args.lang

        # First pass: collect all data by groups
        for month in base_path.iterdir():
            month_num = int(month.name)
            agg_group = (month_num - 1) // self.args.agg_months

            for chunk in month.iterdir():
                for model in chunk.iterdir():
                    gen_file = model / "gen.csv"
                    if not gen_file.exists():
                        continue

                    gen_df = pd.read_csv(gen_file)

                    # Create group key for aggregation
                    for temp in self.args.temp_lst:
                        group_key = (
                            settings.dataset_name_dict[dataset_path.name],
                            chunk.name,
                            model.name,
                            temp,
                            agg_group,
                        )

                        # Get text data for this temperature
                        texts = gen_df[f"unprompted_{temp}"].apply(char2word).tolist()
                        grouped_data[group_key].extend(texts)

        return grouped_data

    def _compute_aggregated_metrics(
        self, data: list[str], word_dict: dict, chunk_size: int, metric_lst: list[str]
    ) -> dict[str, float]:
        """Compute aggregated TTR and rejection rates."""
        # Ensure all data items are strings and handle potential non-string values
        cleaned_data = []
        for item in data:
            if isinstance(item, (float, int)):
                if pd.isna(item):  # Handle NaN values
                    continue
                cleaned_data.append(str(item))
            elif isinstance(item, str):
                if item.strip():  # Only add non-empty strings
                    cleaned_data.append(item)
            else:
                continue  # Skip any other types
        if not cleaned_data:  # If no valid data after cleaning
            return {metric: 0.0 for metric in metric_lst}
        # Aggregate cleaned text data
        aggregated_text = " ".join(cleaned_data)

        # Create metric instance with aggregated data
        metric = Metric(
            data=segment_sent([aggregated_text]),  # Pass as a list for consistency
            chunk_size=chunk_size,
            word_dict=word_dict,
        )

        # Compute requested metrics
        results = {}
        if "type_token_ratio" in metric_lst:
            results["type_token_ratio"] = metric.compute_ttr()
        if "rej_type_rate" in metric_lst:
            results["rej_type_rate"] = metric.compute_type_rej_rate()
        if "rej_token_rate" in metric_lst:
            results["rej_token_rate"] = metric.compute_token_rej_rate()
        return results

    def _compute_monthly_CDI(
        self,
        sent_lst: list[str],
        CDI_words: list[str],
        CDI_words_lst: list[list[str]],
        word_count_est: float,
        previous_words: dict[str, int],
        threshold: int,
    ) -> tuple[float | None, list[float] | None, dict | None, pd.DataFrame | None]:
        """Compute both overall and binned monthly CDI scores if enabled."""
        if not self.CDI_enabled:
            return None, None, None, None

        # Calculate overall CDI score
        calculator = CDICalculator(
            CDI_words=CDI_words,
            word_count_est=word_count_est,
            word_list=segment_sent(sent_lst),
            previous_words=previous_words,
        )
        cum_counts = calculator.get_combined_counts()
        overall_cdi_score = calculator.compute_mean_cdi_score(cum_counts, threshold)
        df = calculator.compute_freq()

        # Calculate binned CDI scores
        binned_scores = []
        for bin_words in CDI_words_lst:
            bin_calculator = CDICalculator(
                CDI_words=bin_words,
                word_count_est=word_count_est,
                word_list=segment_sent(sent_lst),
                previous_words={k: v for k, v in previous_words.items() if k in bin_words},
            )
            bin_counts = bin_calculator.get_combined_counts()
            bin_score = bin_calculator.compute_mean_cdi_score(bin_counts, threshold)
            binned_scores.append(bin_score)

        return overall_cdi_score, binned_scores, cum_counts, df

    def process_model_data(self) -> None:
        """Process model-generated data with month aggregation."""
        for dataset in self.paths["gen_dir"].iterdir():
            if not dataset.is_dir():
                continue

            CDI_words, CDI_words_lst, previous_words = self.load_CDI_words(dataset.name)
            word_dict = load_dict(settings.dataset_name_dict[dataset.name])
            base_path = dataset / f"{self.args.hour_per_year}_hour_per_year" / self.args.lang
            freq_path = (
                self.paths["freq_dir"] / dataset.name / f"{self.args.hour_per_year}_hour_per_year" / self.args.lang
            )

            # Create expanded metric list with binned CDI scores
            metric_columns = self.args.metric_lst.copy()
            if "CDI" in metric_columns:
                metric_columns.extend([f"CDI_{i}" for i in range(self.n_bins)])

            # First compute aggregated metrics if needed
            agg_metrics = {}
            if self.args.agg_months > 1:
                for month in base_path.iterdir():
                    month_num = int(month.name)
                    agg_group = (month_num - 1) // self.args.agg_months

                    for chunk in month.iterdir():
                        for model in chunk.iterdir():
                            if not (model / "gen.csv").exists():
                                continue

                            gen_df = pd.read_csv(model / "gen.csv")
                            group_key = (agg_group, chunk.name, model.name)

                            if group_key not in agg_metrics:
                                agg_metrics[group_key] = defaultdict(list)

                            # Collect texts for aggregated non-CDI metrics
                            for temp in self.args.temp_lst:
                                texts = gen_df[f"unprompted_{temp}"].apply(char2word).tolist()
                                agg_metrics[group_key][temp].extend(texts)

                # Compute aggregated metrics once per group
                for group_key, texts_by_temp in agg_metrics.items():
                    agg_group, chunk, model = group_key
                    for temp, texts in texts_by_temp.items():
                        non_cdi_metrics = [m for m in self.args.metric_lst if m in self.non_cdi_metrics]
                        if non_cdi_metrics:
                            agg_metrics[group_key][temp] = self._compute_aggregated_metrics(
                                texts, word_dict, self.args.chunk_size, non_cdi_metrics
                            )

            # Process each month
            for month in base_path.iterdir():
                month_num = int(month.name)
                agg_group = (month_num - 1) // self.args.agg_months if self.args.agg_months > 1 else None

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

                        scores = []
                        freq_file_path = freq_path / month.name / chunk.name / model.name
                        freq_file_path.mkdir(parents=True, exist_ok=True)
                        for temp in self.args.temp_lst:
                            try:
                                monthly_texts = gen[f"unprompted_{temp}"].apply(char2word).tolist()
                                word_dict_manager = WordDictManager(
                                    dataset=info_dict["dataset"],
                                    chunk=info_dict["chunk"],
                                    model_type=info_dict["model_type"],
                                    temp=temp,
                                )
                                previous_words = word_dict_manager.load_word_dict(self.CDI_month_dict)

                                overall_cdi, binned_cdi, cum_counts, df = self._compute_monthly_CDI(
                                    monthly_texts,
                                    CDI_words,
                                    CDI_words_lst,
                                    self.load_word_count_est(month_num),
                                    previous_words,
                                    self.threshold,
                                )

                                if agg_group is not None and (agg_group, chunk.name, model.name) in agg_metrics:
                                    non_cdi_metrics = agg_metrics[(agg_group, chunk.name, model.name)][temp]
                                else:
                                    non_cdi_metrics = self._compute_aggregated_metrics(
                                        monthly_texts,
                                        word_dict,
                                        self.args.chunk_size,
                                        [m for m in self.args.metric_lst if m in self.non_cdi_metrics],
                                    )

                                # Create row with metrics
                                row = [month_num]
                                for metric_name in self.args.metric_lst:
                                    if metric_name == "CDI":
                                        row.append(overall_cdi)
                                    else:
                                        row.append(non_cdi_metrics.get(metric_name))

                                # Add binned CDI scores if CDI is enabled
                                if "CDI" in self.args.metric_lst and binned_cdi is not None:
                                    row.extend(binned_cdi)

                                row.append(sum(len(text.split()) for text in monthly_texts))
                                row.append(temp)
                                scores.append(row)

                                if cum_counts is not None:
                                    self.CDI_month_dict = word_dict_manager.write_word_dict(
                                        cum_counts, self.CDI_month_dict
                                    )

                                # Save frequency data if available
                                if df is not None:
                                    df.to_csv(freq_file_path / f"freq_{temp}.csv", index=False)

                            except Exception as e:
                                print(f"Error processing temperature {temp}: {e}")
                                continue

                        if scores:
                            columns = ["month"] + metric_columns + ["word_num", "temp"]
                            score = pd.DataFrame(scores, columns=columns)
                            score = score.assign(**info_dict)
                            self.score_all = pd.concat([self.score_all, score])

    def process_human_data(self) -> None:
        """Process human reference data and compute metrics."""
        ref_data = pd.read_csv(self.paths["ref_dir"])
        CDI_words, CDI_words_lst, previous_words = self.load_CDI_words("CHILDES")
        word_dict = load_dict("CHILDES")

        score_human = self._compute_human_metrics(
            ref_data=ref_data, CDI_words=CDI_words, CDI_words_lst=CDI_words_lst, word_dict=word_dict
        )

        if not score_human.empty:
            score_human = score_human[self.score_all.columns]
            self.score_all = pd.concat([score_human, self.score_all])

    def _compute_human_metrics(
        self, ref_data: pd.DataFrame, CDI_words: list[str], CDI_words_lst: list[list[str]], word_dict: dict
    ) -> pd.DataFrame:
        """Compute metrics for human reference data."""
        info_dict = {"dataset": "CHILDES", "chunk": "00", "model_type": "human", "temp": "1.0"}
        scores = []
        gen_grouped = ref_data.groupby("month")

        # Initialize aggregation metrics dictionary outside the loop
        agg_metrics = {}

        # Create expanded metric list with binned CDI scores
        metric_columns = self.args.metric_lst.copy()
        if "CDI" in metric_columns:
            metric_columns.extend([f"CDI_{i}" for i in range(self.n_bins)])

        # Compute aggregated metrics for each group if needed
        if self.args.agg_months > 1:
            for month in gen_grouped.groups:
                agg_group = (month - 1) // self.args.agg_months
                if agg_group not in agg_metrics:
                    group_months = range(
                        agg_group * self.args.agg_months + 1, (agg_group + 1) * self.args.agg_months + 1
                    )
                    # Collect all text data for this group
                    agg_data = []
                    for m in group_months:
                        if m in gen_grouped.groups:
                            month_data = gen_grouped.get_group(m)["text"].fillna("").astype(str).tolist()
                            agg_data.extend(month_data)

                    # Compute aggregated metrics
                    if agg_data:
                        agg_metrics[agg_group] = self._compute_aggregated_metrics(
                            agg_data,
                            word_dict,
                            self.args.chunk_size,
                            [m for m in self.args.metric_lst if m in self.non_cdi_metrics],
                        )

        # Process each month
        for month, gen in gen_grouped:
            word_dict_manager = WordDictManager(
                dataset=info_dict["dataset"],
                chunk=info_dict["chunk"],
                model_type=info_dict["model_type"],
                temp=info_dict["temp"],
            )

            word_count_est = self.load_word_count_est(month)
            previous_words = word_dict_manager.load_word_dict(self.CDI_month_dict)
            sent_lst = gen["text"].fillna("").astype(str).tolist()

            # Get overall and binned CDI scores
            overall_cdi, binned_cdi, cum_counts, df = self._compute_monthly_CDI(
                sent_lst, CDI_words, CDI_words_lst, word_count_est, previous_words, self.threshold
            )

            # Create row with metrics
            row = [month]
            agg_group = (month - 1) // self.args.agg_months if self.args.agg_months > 1 else None

            for metric_name in self.args.metric_lst:
                if metric_name == "CDI":
                    row.append(overall_cdi)
                elif self.args.agg_months > 1 and agg_group in agg_metrics:
                    row.append(agg_metrics[agg_group].get(metric_name))
                else:
                    curr_metrics = self._compute_aggregated_metrics(
                        sent_lst, word_dict, self.args.chunk_size, [metric_name]
                    )
                    row.append(curr_metrics.get(metric_name))

            # Add binned CDI scores if CDI is enabled
            if "CDI" in self.args.metric_lst and binned_cdi is not None:
                row.extend(binned_cdi)

            row.append(gen["sent_len"].sum())  # word_num
            row.append(info_dict["temp"])  # temperature
            scores.append(row)

            if cum_counts is not None:
                self.CDI_month_dict = word_dict_manager.write_word_dict(cum_counts, self.CDI_month_dict)

        if not scores:
            return pd.DataFrame()

        columns = ["month"] + metric_columns + ["word_num", "temp"]
        score = pd.DataFrame(scores, columns=columns)
        score = score.assign(**info_dict)
        return score

    def save_results(self) -> None:
        """Save computed metrics to file."""
        # save the metric into the same directory as the CDI files
        if self.CDI_enabled:
            CDIdataset = WordStatsDataset(sampling_ratio=self.sampling_ratio)
            output_dir = CDIdataset.matched_root
        else:
            output_dir = self.paths["metric_dir"] 
        output_path = output_dir / f"metric_{self.args.hour_per_year}_{self.args.agg_months}_{self.args.chunk_size}.csv"
        self.score_all.to_csv(output_path)
        print(f"Saving the metric to {output_path}")

    def run(self) -> None:
        """Run the complete metrics computation pipeline."""
        self.process_model_data()
        self.process_human_data()
        self.save_results()


def main() -> None:
    """Main entry point for metrics computation."""
    args = parse_args()
    computer = MetricsProcessor(args)
    computer.run()


if __name__ == "__main__":
    main()
