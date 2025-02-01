"""Test CDI threhsolds."""
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lexical_benchmark import settings
from lexical_benchmark.stats.CDI_scores import CDICalculator
from lexical_benchmark.stats.metric import CDICalculator

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
        default="gen/merged/",
        help="relative path to save metrics",
    )
    parser.add_argument(
        "--CDI_path",
        type=str,
        default="metrics/material/",
        help="relative path to CDI scores"
    )
    parser.add_argument(
        "--word_est_path",
        type=str,
        default="metrics/material/vocal_month.csv",
        help="relative path to vocal estimation"
    )
    parser.add_argument("--threshold_list", default=60, type=int, help="threshold to compute CDI scores")
    return parser.parse_args()

class MetricsProcessor:
    """Compute and manage metrics for both model-generated and human data."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the MetricsProcessor with command line arguments."""
        self.args = args
        self.paths = self._setup_paths()
        self.word_est_dict = self._load_word_est_dict()
        self.CDI_month_dict: dict[str, dict] = {"dataset": {"chunk": {"model_type": {"temp": {}}}}}
        self.score_all = pd.DataFrame()
        self.non_cdi_metrics = ["type_token_ratio", "rej_type_rate", "rej_token_rate"]

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
        CDI_frame = pd.read_csv(self.paths["CDI_dir"] / f"{dataset}_CDI.csv")
        CDI_words = CDI_frame["word"].tolist()
        return CDI_words, dict.fromkeys(CDI_words, 0)

    def load_word_count_est(self, month: int) -> float:
        """Load word count estimation for each month."""
        return self.word_est_dict.get(month, 0) if self.CDI_enabled else 0

    def _compute_monthly_CDI(
        self,
        sent_lst: list[str],
        temp: float | str,
        CDI_words: list[str],
        word_dict: dict,
        word_count_est: float,
        previous_words: dict[str, int],
    ) -> tuple[float | None, dict | None]:
        """Compute monthly CDI scores if enabled."""
        if not self.CDI_enabled:
            return None, None

        metric = Metric(
            data=sent_lst,
            temp=temp,
            metric_lst=["CDI"],
            threshold=self.args.threshold,
            CDI_words=CDI_words,
            word_count_est=word_count_est,
            chunk_size=self.args.chunk_size,
            word_dict=word_dict,
            previous_words=previous_words,
        )

        cdi_results, cum_counts = metric.compute_metrics()
        cdi_score = cdi_results[4] if cdi_results else None

        return cdi_score, cum_counts

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
        scores = []
        gen_grouped = ref_data.groupby("month")

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
            # Convert to string and handle NaN values
            sent_lst = gen["text"].fillna("").astype(str).tolist()

            # Get CDI score if enabled
            cdi_score, cum_counts = self._compute_monthly_CDI(
                sent_lst, month, CDI_words, word_dict, word_count_est, previous_words
            )

            # Create row with metrics
            row = [month]
            agg_group = (month - 1) // self.args.agg_months if self.args.agg_months > 1 else None

            for metric_name in self.args.metric_lst:
                if metric_name == "CDI":
                    row.append(cdi_score)
                elif self.args.agg_months > 1 and agg_group in agg_metrics:
                    row.append(agg_metrics[agg_group].get(metric_name))
                else:
                    # Compute non-aggregated metrics for single month
                    curr_metrics = self._compute_aggregated_metrics(
                        sent_lst, word_dict, self.args.chunk_size, [metric_name]
                    )
                    row.append(curr_metrics.get(metric_name))

            row.append(gen["sent_len"].sum())
            scores.append(row)

            if cum_counts is not None:
                self.CDI_month_dict = word_dict_manager.write_word_dict(cum_counts, self.CDI_month_dict)
        if not scores:
            return pd.DataFrame()

        columns = ["month"] + self.args.metric_lst + ["word_num"]
        score = pd.DataFrame(scores, columns=columns)
        score = score.assign(**info_dict)
        return score

    def save_results(self) -> None:
        """Save computed metrics to file."""
        output_path = self.paths["metric_dir"] / f"metric_{self.args.hour_per_year}_{self.args.agg_months}.csv"
        self.score_all.to_csv(output_path)
        print(f"Saving the metric to {output_path}")

    def run(self) -> None:
        """Run the complete metrics computation pipeline."""
        self.process_model_data()
        self.process_human_data()
        self.save_results()


def main():
    # Args parser
    args = parse_args()
    # load paths
    metric_dir: Path = settings.PATH.DATA_DIR / args.metric_path
    ref_dir: Path = settings.PATH.DATA_DIR / args.ref_path
    CDI_dir: Path = settings.PATH.DATA_DIR / args.CDI_path
    word_est_dir: Path = settings.PATH.DATA_DIR / args.word_est_path
    # load monthly estimation dict
    df_est = pd.read_csv(word_est_dir)
    word_est_dict = dict(zip(df_est["month"], df_est["child_month_est"]))
    print(f"Monthly production estimation loaded {word_est_dict}")
    # compute human production
    ref_data = pd.read_csv(ref_dir)
    CDI_frame = pd.read_csv(CDI_path)
    CDI_words = CDI_frame['word'].tolist()

    
    for threshold in args.threshold_list:



    score_human = append_human_metric(ref_data,args.metric_lst,args.threshold,,args.chunk_size, 
        word_dict,word_est_dict,CDI)
    # Reorder columns based on given list
    score_human = score_human[score_all.columns]
    score_all = pd.concat([score_human,score_all])
    score_all.to_csv(metric_dir/f"metric_{args.hour_per_year}.csv")
    print(f"Saving the metric to {metric_dir}/metric_{args.hour_per_year}.csv")

if __name__ == "__main__":
    main()
