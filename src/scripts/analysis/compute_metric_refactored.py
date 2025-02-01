#!/usr/bin/env python
"""Compute core metrics from the generation directory."""

import argparse
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lexical_benchmark import settings
from lexical_benchmark.datasets.utils.text_cleaning import char2word
from lexical_benchmark.stats.metric import Metric, WordDictManager, load_dict


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="compute metrics")
    parser.add_argument("--gen_path", type=str, default="gen/merged", help="relative path to the generated texts")
    parser.add_argument(
        "--ref_path", type=str, default="gen/merged/CHILDES_model.csv", help="relative path to the human reference data"
    )
    parser.add_argument("--metric_path", type=str, default="gen/merged/", help="relative path to save metrics")
    parser.add_argument("--CDI_path", type=str, default="metrics/material/", help="relative path to CDI scores")
    parser.add_argument(
        "--word_est_path",
        type=str,
        default="metrics/material/vocal_month.csv",
        help="relative path to vocal estimation",
    )
    parser.add_argument(
        "--metric_lst", type=list, default=["type_token_ratio", "rej_type_rate", "rej_token_rate", "CDI"]
    )
    parser.add_argument("--temp_lst", type=list, default=[1.0])
    parser.add_argument("--hour_per_year", default=1000, type=int)
    parser.add_argument("--chunk_size", default=1000, type=int)
    parser.add_argument("--threshold", default=1, type=int)
    parser.add_argument("--lang", default="EN", type=str)
    parser.add_argument("--agg_months", type=int, default=2)
    return parser.parse_args()


@dataclass
class MetricConfig:
    """Configuration for metric computation."""

    gen_path: Path
    metric_path: Path
    ref_path: Path
    CDI_path: Path
    word_est_path: Path
    metric_lst: list[str]
    temp_lst: list[float]
    hour_per_year: int
    chunk_size: int
    threshold: int
    lang: str
    agg_months: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "MetricConfig":
        """Create configuration from command line arguments."""
        paths = {
            "gen_path": settings.PATH.DATA_DIR / args.gen_path,
            "metric_path": settings.PATH.DATA_DIR / args.metric_path,
            "ref_path": settings.PATH.DATA_DIR / args.ref_path,
            "CDI_path": settings.PATH.DATA_DIR / args.CDI_path,
            "word_est_path": settings.PATH.DATA_DIR / args.word_est_path,
        }
        return cls(
            **paths,
            metric_lst=args.metric_lst,
            temp_lst=args.temp_lst,
            hour_per_year=args.hour_per_year,
            chunk_size=args.chunk_size,
            threshold=args.threshold,
            lang=args.lang,
            agg_months=args.agg_months,
        )


class MetricsComputer:
    """Core metrics computation class."""

    def __init__(self, config: MetricConfig):
        """Initialize with configuration."""
        self.config = config
        self.word_dict: dict[str, t.Any] = {}
        self.CDI_words: list[str] = []
        self.previous_words: dict[str, int] = {}
        self.CDI_month_dict: dict[str, dict] = {"dataset": {"chunk": {"model_type": {"temp": {}}}}}

    def compute_metrics(self, texts: list[str], word_dict: dict) -> dict[str, float]:
        """Compute all non-CDI metrics for a set of texts."""
        if not texts:
            return {metric: 0.0 for metric in self.config.metric_lst}

        metric = Metric(data=texts, chunk_size=self.config.chunk_size, word_dict=word_dict)

        results = {}
        for metric_name in self.config.metric_lst:
            if metric_name == "type_token_ratio":
                results[metric_name] = metric.compute_ttr()
            elif metric_name == "rej_type_rate":
                results[metric_name] = metric.compute_type_rej_rate()
            elif metric_name == "rej_token_rate":
                results[metric_name] = metric.compute_token_rej_rate()
        return results

    def compute_CDI(
        self, texts: list[str], temp: float | str, word_count_est: float, word_dict_manager: WordDictManager
    ) -> tuple[float | None, dict | None]:
        """Compute CDI scores and update cumulative word counts."""
        if "CDI" not in self.config.metric_lst:
            return None, None

        previous_words = word_dict_manager.load_word_dict(self.CDI_month_dict)
        metric = Metric(
            data=texts,
            temp=temp,
            metric_lst=["CDI"],
            threshold=self.config.threshold,
            CDI_words=self.CDI_words,
            word_count_est=word_count_est,
            chunk_size=self.config.chunk_size,
            word_dict=self.word_dict,
            previous_words=previous_words,
        )

        cdi_results, cum_counts = metric.compute_metrics()
        if cum_counts is not None:
            self.CDI_month_dict = word_dict_manager.write_word_dict(cum_counts, self.CDI_month_dict)
        return cdi_results[4] if cdi_results else None, cum_counts


class DataProcessor:
    """Process and aggregate data for metric computation."""

    def __init__(self, config: MetricConfig, metrics_computer: MetricsComputer):
        self.config = config
        self.metrics_computer = metrics_computer
        self.word_est_dict = self._load_word_est_dict()

    def _load_word_est_dict(self) -> dict[int, float]:
        """Load word estimation dictionary."""
        if "CDI" not in self.config.metric_lst:
            return {}
        df_est = pd.read_csv(self.config.word_est_path)
        return dict(zip(df_est["month"], df_est["child_month_est"]))

    def aggregate_texts(self, texts: list[str | float]) -> list[str]:
        """Clean and aggregate text data."""
        cleaned_texts = []
        for item in texts:
            if isinstance(item, (float, int)):
                if not pd.isna(item):
                    cleaned_texts.append(str(item))
            elif isinstance(item, str) and item.strip():
                cleaned_texts.append(item)
        return cleaned_texts


class MetricsProcessor:
    """Main class to orchestrate metrics processing."""

    def __init__(self, args: argparse.Namespace):
        self.config = MetricConfig.from_args(args)
        self.metrics_computer = MetricsComputer(self.config)
        self.data_processor = DataProcessor(self.config, self.metrics_computer)
        self.score_all = pd.DataFrame()

    def process_dataset(self, dataset_path: Path, is_human: bool = False) -> None:
        """Process a single dataset directory."""
        dataset_name = settings.dataset_name_dict[dataset_path.name]
        self.metrics_computer.word_dict = load_dict(dataset_name)

        if "CDI" in self.config.metric_lst:
            CDI_frame = pd.read_csv(self.config.CDI_path / f"{dataset_path.name}_CDI.csv")
            self.metrics_computer.CDI_words = CDI_frame["word"].tolist()
            self.metrics_computer.previous_words = dict.fromkeys(self.metrics_computer.CDI_words, 0)

        if is_human:
            self._process_human_data(dataset_name)
        else:
            self._process_model_data(dataset_path, dataset_name)

    def _collect_model_data(self, base_path: Path) -> dict[tuple, dict[str, list[str]]]:
        """Collect model data across months for aggregation."""
        collected_data = defaultdict(lambda: defaultdict(list))

        for month in sorted(base_path.iterdir()):
            if not month.is_dir():
                continue

            month_num = int(month.name)
            agg_group = (month_num - 1) // self.config.agg_months if self.config.agg_months > 1 else month_num

            for chunk in month.iterdir():
                for model in chunk.iterdir():
                    gen_file = model / "gen.csv"
                    if not gen_file.exists():
                        continue

                    gen_df = pd.read_csv(gen_file)
                    group_key = (chunk.name, model.name, agg_group)

                    # Collect texts for each temperature
                    for temp in self.config.temp_lst:
                        col_name = f"unprompted_{temp}"
                        if col_name in gen_df.columns:
                            texts = gen_df[col_name].apply(char2word).tolist()
                            texts = self.data_processor.aggregate_texts(texts)
                            collected_data[group_key][str(temp)].extend(texts)

                    # Store month number for CDI computation
                    collected_data[group_key]["months"].append(month_num)

        return collected_data

    def _process_model_data(self, dataset_path: Path, dataset_name: str) -> None:
        """Process model data with proper month aggregation."""
        base_path = dataset_path / f"{self.config.hour_per_year}_hour_per_year" / self.config.lang
        collected_data = self._collect_model_data(base_path)

        for (chunk, model_type, agg_group), data in collected_data.items():
            scores = []

            for temp in self.config.temp_lst:
                temp_str = str(temp)
                if temp_str not in data:
                    continue

                try:
                    # Create WordDictManager for this temperature
                    word_dict_manager = WordDictManager(
                        dataset=dataset_name, chunk=chunk, model_type=model_type, temp=temp
                    )

                    # Compute metrics on aggregated texts
                    texts = data[temp_str]
                    metrics = self.metrics_computer.compute_metrics(texts, self.metrics_computer.word_dict)

                    # Compute CDI for each month in the group
                    months = sorted(data["months"])
                    for month in months:
                        month_texts = []  # Get month-specific texts for CDI
                        word_count_est = self.data_processor.word_est_dict.get(month, 0)

                        cdi_score, _ = self.metrics_computer.compute_CDI(
                            texts=month_texts if month_texts else texts,
                            temp=temp,
                            word_count_est=word_count_est,
                            word_dict_manager=word_dict_manager,
                        )

                        # Build row with metrics
                        row = [month]
                        for metric_name in self.config.metric_lst:
                            if metric_name == "CDI":
                                row.append(cdi_score)
                            else:
                                row.append(metrics.get(metric_name))
                        row.append(sum(len(text.split()) for text in texts))
                        scores.append(row)

                except Exception as e:
                    print(f"Error processing model data: {e}")
                    continue

            if scores:
                columns = ["month"] + self.config.metric_lst + ["word_num"]
                score_df = pd.DataFrame(scores, columns=columns)
                score_df = score_df.assign(dataset=dataset_name, chunk=chunk, model_type=model_type)
                self.score_all = pd.concat([self.score_all, score_df])

    def _process_human_data(self, dataset_name: str) -> None:
        """Process human reference data with proper aggregation."""
        if not self.config.ref_path.exists():
            return

        ref_data = pd.read_csv(self.config.ref_path)
        gen_grouped = ref_data.groupby("month")

        # Initialize aggregation metrics if needed
        agg_metrics = {}
        if self.config.agg_months > 1:
            for month in gen_grouped.groups:
                agg_group = (month - 1) // self.config.agg_months
                if agg_group not in agg_metrics:
                    group_months = range(
                        agg_group * self.config.agg_months + 1, (agg_group + 1) * self.config.agg_months + 1
                    )
                    agg_data = []
                    for m in group_months:
                        if m in gen_grouped.groups:
                            month_data = gen_grouped.get_group(m)["text"].fillna("").astype(str).tolist()
                            agg_data.extend(month_data)

                    if agg_data:
                        agg_metrics[agg_group] = self.metrics_computer.compute_metrics(
                            agg_data, self.metrics_computer.word_dict
                        )

        # Process each month
        scores = []
        for month, gen in gen_grouped:
            word_dict_manager = WordDictManager(dataset=dataset_name, chunk="00", model_type="human", temp="1.0")

            word_count_est = self.data_processor.word_est_dict.get(month, 0)
            sent_lst = gen["text"].fillna("").astype(str).tolist()

            # Get CDI score
            cdi_score, _ = self.metrics_computer.compute_CDI(sent_lst, "1.0", word_count_est, word_dict_manager)

            # Build row with metrics
            row = [month]
            agg_group = (month - 1) // self.config.agg_months if self.config.agg_months > 1 else None

            for metric_name in self.config.metric_lst:
                if metric_name == "CDI":
                    row.append(cdi_score)
                elif self.config.agg_months > 1 and agg_group in agg_metrics:
                    row.append(agg_metrics[agg_group].get(metric_name))
                else:
                    curr_metrics = self.metrics_computer.compute_metrics(sent_lst, self.metrics_computer.word_dict)
                    row.append(curr_metrics.get(metric_name))

            row.append(gen["sent_len"].sum())
            scores.append(row)

        if scores:
            columns = ["month"] + self.config.metric_lst + ["word_num"]
            score_df = pd.DataFrame(scores, columns=columns)
            score_df = score_df.assign(dataset=dataset_name, chunk="00", model_type="human", temp="1.0")
            self.score_all = pd.concat([self.score_all, score_df])

    def run(self) -> None:
        """Run the complete metrics computation pipeline."""
        # Process model data
        for dataset in self.config.gen_path.iterdir():
            if dataset.is_dir():
                self.process_dataset(dataset)

        # Process human data
        self.process_human_data()

        # Save results
        output_path = self.config.metric_path / f"metric_{self.config.hour_per_year}_{self.config.agg_months}.csv"
        self.score_all.to_csv(output_path)
        print(f"Metrics saved to {output_path}")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    processor = MetricsProcessor(args)
    processor.run()


if __name__ == "__main__":
    main()
