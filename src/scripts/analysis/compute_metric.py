#!/usr/bin/env python
"""Compute core metrics from the generation directory."""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lexical_benchmark import settings
from lexical_benchmark.datasets.utils.text_cleaning import char2word
from lexical_benchmark.stats.metric import (
    Metric,
    WordDictManager,
    load_dict,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="compute metrics")
    parser.add_argument(
        "--gen_path",
        type=str,
        default="gen/merged",
        help="relative path to the generated texts",
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        default="gen/merged/CHILDES_model.csv",
        help="relative path to the human reference data",
    )
    parser.add_argument(
        "--metric_path",
        type=str,
        default="gen/merged/",
        help="relative path to save metrics",
    )
    parser.add_argument("--CDI_path", type=str, default="metrics/material/", help="relative path to CDI scores")
    parser.add_argument(
        "--word_est_path",
        type=str,
        default="metrics/material/vocal_month.csv",
        help="relative path to vocal estimation",
    )
    parser.add_argument(
        "--metric_lst",
        type=list,
        default=["type_token_ratio", "rej_type_rate", "rej_token_rate", "CDI"],
        help="metric list; ttr,rej_type_rate,CDI",
    )
    parser.add_argument("--temp_lst", type=list, default=[0.3, 0.6, 1.0, 1.5], help="temperature list")
    parser.add_argument("--hour_per_year", default=1000, type=int, help="Estimated yearly exposure hours")
    parser.add_argument("--chunk_size", default=500, type=int, help="Chunk size to normalize the scores")
    parser.add_argument("--threshold", default=1, type=int, help="threshold to compute CDI scores")
    parser.add_argument("--lang", default="EN", type=str, help="tested language")
    return parser.parse_args()


class MetricsProcessor:
    """Compute and manage metrics for both model-generated and human data."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the MetricsComputer with command line arguments."""
        self.args = args
        self.paths = self._setup_paths()
        self.word_est_dict = self._load_word_est_dict()
        self.CDI_enabled = "CDI" in args.metric_lst
        self.CDI_month_dict: dict[str, dict] = {"dataset": {"chunk": {"model_type": {"temp": {}}}}}
        self.score_all = pd.DataFrame()
        self.metric_indices = {"type_token_ratio": 1, "rej_type_rate": 2, "rej_token_rate": 3, "CDI": 4}

    def _setup_paths(self) -> dict[str, Path]:
        """Set up all necessary paths."""
        return {
            "gen_dir": settings.PATH.DATA_DIR / self.args.gen_path,
            "metric_dir": settings.PATH.DATA_DIR / self.args.metric_path,
            "ref_dir": settings.PATH.DATA_DIR / self.args.ref_path,
            "CDI_dir": settings.PATH.DATA_DIR / self.args.CDI_path,
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
            CDI_frame = pd.read_csv(self.paths["CDI_dir"] / f"{dataset}_CDI.csv")
            CDI_words = CDI_frame["word"].tolist()
            return CDI_words, dict.fromkeys(CDI_words, 0)
        return [], {}

    def load_word_count_est(self, month: int) -> float:
        """Load word count estimation for each month."""
        return self.word_est_dict.get(month, 0) if self.CDI_enabled else 0

    def _compute_model_metrics(
        self, gen: pd.DataFrame, info_dict: dict[str, str], CDI_words: list[str], word_dict: dict, word_count_est: float
    ) -> tuple[pd.DataFrame, dict]:
        """Compute metrics for model-generated data."""
        scores = []

        for temp in self.args.temp_lst:
            word_dict_manager = WordDictManager(
                dataset=info_dict["dataset"], chunk=info_dict["chunk"], model_type=info_dict["model_type"], temp=temp
            )

            previous_words = word_dict_manager.load_word_dict(self.CDI_month_dict)
            sent_lst = gen[f"unprompted_{temp}"].apply(char2word).tolist()

            metric = Metric(
                data=sent_lst,
                temp=temp,
                metric_lst=self.args.metric_lst,
                threshold=self.args.threshold,
                CDI_words=CDI_words,
                word_count_est=word_count_est,
                chunk_size=self.args.chunk_size,
                word_dict=word_dict,
                previous_words=previous_words,
            )

            metric_results, cum_counts = metric.compute_metrics()

            if metric_results:
                scores.append(metric_results)

            if cum_counts is not None:
                self.CDI_month_dict = word_dict_manager.write_word_dict(cum_counts, self.CDI_month_dict)

        if not scores:
            return pd.DataFrame(), self.CDI_month_dict

        score = pd.DataFrame(scores, columns=["temp"] + self.args.metric_lst)
        score = score.assign(**info_dict)
        score["word_num"] = gen["sent_len"].sum()

        return score, self.CDI_month_dict

    def process_model_data(self) -> None:
        """Process model-generated data and compute metrics."""
        for dataset in self.paths["gen_dir"].iterdir():
            if not dataset.is_dir():
                continue

            CDI_words, previous_words = self.load_CDI_words(dataset.name)
            word_dict = load_dict(settings.dataset_name_dict[dataset.name])

            base_path = dataset / f"{self.args.hour_per_year}_hour_per_year" / self.args.lang

            for month in tqdm(base_path.iterdir()):
                word_count_est = self.load_word_count_est(int(month.name))
                self._process_month(month, dataset, CDI_words, word_dict, word_count_est)

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

        score_human = self._compute_human_metrics(ref_data=ref_data, CDI_words=CDI_words, word_dict=word_dict)

        if not score_human.empty:
            score_human = score_human[self.score_all.columns]
            self.score_all = pd.concat([score_human, self.score_all])

    def _compute_human_metrics(self, ref_data: pd.DataFrame, CDI_words: list[str], word_dict: dict) -> pd.DataFrame:
        """Compute metrics for human reference data."""
        info_dict = {"dataset": "CHILDES", "chunk": "00", "model_type": "human", "temp": "1.0"}

        gen_grouped = ref_data.groupby("month")
        scores = []

        for month, gen in gen_grouped:
            word_dict_manager = WordDictManager(
                dataset=info_dict["dataset"],
                chunk=info_dict["chunk"],
                model_type=info_dict["model_type"],
                temp=info_dict["temp"],
            )

            word_count_est = self.load_word_count_est(month)
            previous_words = word_dict_manager.load_word_dict(self.CDI_month_dict)
            sent_lst = gen["text"].tolist()

            metric = Metric(
                data=sent_lst,
                temp=month,
                metric_lst=self.args.metric_lst,
                threshold=self.args.threshold,
                CDI_words=CDI_words,
                word_count_est=word_count_est,
                chunk_size=self.args.chunk_size,
                word_dict=word_dict,
                previous_words=previous_words,
            )

            metric_results, cum_counts = metric.compute_metrics()

            if metric_results:
                row = metric_results
                row.append(gen["sent_len"].sum())
                scores.append(row)

            if cum_counts is not None:
                self.CDI_month_dict = word_dict_manager.write_word_dict(cum_counts, self.CDI_month_dict)

        if not scores:
            return pd.DataFrame()

        score = pd.DataFrame(scores, columns=["month"] + self.args.metric_lst + ["word_num"])
        score = score.assign(**info_dict)

        return score

    def save_results(self) -> None:
        """Save computed metrics to file."""
        output_path = self.paths["metric_dir"] / f"metric_{self.args.hour_per_year}.csv"
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
