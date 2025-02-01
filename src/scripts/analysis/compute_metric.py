#!/usr/bin/env python
"""Compute core metrics from the generation directory."""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lexical_benchmark import settings
from lexical_benchmark.datasets.utils.text_cleaning import char2word
from lexical_benchmark.stats.metric import Metric, WordDictManager, load_dict


def parse_args():
    # Run parameters
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


def compute_metrics_base(
    sent_lst: list[str],
    temp: str | float,
    metric_lst: list[str],
    threshold: int,
    CDI_words: list[str],
    chunk_size: int,
    word_dict: dict,
    word_count_est: int,
    previous_words: dict[str, int],
    CDI_month_dict: dict[str, dict],
) -> tuple[list, dict[str, dict]]:
    """Base function for computing metrics."""
    # Create metric instance
    metric = Metric(
        data=sent_lst,
        temp=temp,
        metric_lst=metric_lst,
        threshold=threshold,
        CDI_words=CDI_words,
        word_count_est=word_count_est,
        chunk_size=chunk_size,
        word_dict=word_dict,
        previous_words=previous_words,
    )
    return metric.compute_metrics()


def append_model_metric(
    gen: pd.DataFrame,
    temp_lst: list[float],
    info_dict: dict[str, str | int],
    metric_lst: list[str],
    threshold: int,
    CDI_words: list[str],
    chunk_size: int,
    word_dict: dict,
    word_count_est: int,
    CDI_month_dict: dict[str, dict],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Compute model-specific metrics."""
    scores = []
    metric_indices = {"type_token_ratio": 1, "rej_type_rate": 2, "rej_token_rate": 3, "CDI": 4}

    for temp in temp_lst:
        word_dict_manager = WordDictManager(
            dataset=info_dict["dataset"],
            chunk=info_dict["chunk"],
            model_type=info_dict["model_type"],
            temp=temp,
        )

        previous_words = word_dict_manager.load_word_dict(CDI_month_dict)
        sent_lst = gen[f"unprompted_{temp}"].apply(char2word).tolist()

        # Compute metrics using base function
        metric_results, cum_counts = compute_metrics_base(
            sent_lst=sent_lst,
            temp=temp,
            metric_lst=metric_lst,
            threshold=threshold,
            CDI_words=CDI_words,
            word_count_est=word_count_est,
            chunk_size=chunk_size,
            word_dict=word_dict,
            previous_words=previous_words,
            CDI_month_dict=CDI_month_dict,
        )

        if metric_results:
            row = [temp]  # Start with temperature
            for metric_name in metric_lst:
                idx = metric_indices.get(metric_name)
                if idx is not None and idx < len(metric_results):
                    row.append(metric_results[idx])
                else:
                    row.append(None)
            scores.append(row)

        # Update CDI dictionary
        if cum_counts is not None:
            CDI_month_dict = word_dict_manager.write_word_dict(cum_counts, CDI_month_dict)

    if not scores:
        return pd.DataFrame(), CDI_month_dict

    score = pd.DataFrame(scores, columns=["temp"] + metric_lst)
    score = score.assign(**info_dict)
    score["word_num"] = gen["sent_len"].sum()

    return score, CDI_month_dict


def append_human_metric(
    ref_data: pd.DataFrame,
    metric_lst: list[str],
    threshold: int,
    CDI_words: list[str],
    chunk_size: int,
    word_dict: dict,
    word_est_dict: dict[int, float],
    CDI: bool,
    previous_words: dict[str, int],
    CDI_month_dict: dict[str, dict],
) -> pd.DataFrame:
    """Compute human metrics."""
    info_dict = {"dataset": "CHILDES", "chunk": "00", "model_type": "human", "temp": "1.0"}

    gen_grouped = ref_data.groupby("month")
    scores = []
    metric_indices = {"type_token_ratio": 1, "rej_type_rate": 2, "rej_token_rate": 3, "CDI": 4}

    for month, gen in gen_grouped:
        word_dict_manager = WordDictManager(
            dataset=info_dict["dataset"],
            chunk=info_dict["chunk"],
            model_type=info_dict["model_type"],
            temp=info_dict["temp"],
        )

        word_count_est = load_word_count_est(word_est_dict, month, CDI)
        previous_words = word_dict_manager.load_word_dict(CDI_month_dict)
        sent_lst = gen["text"].tolist()

        # Compute metrics using base function
        metric_results, cum_counts = compute_metrics_base(
            sent_lst=sent_lst,
            temp=month,
            metric_lst=metric_lst,
            threshold=threshold,
            CDI_words=CDI_words,
            word_count_est=word_count_est,
            chunk_size=chunk_size,
            word_dict=word_dict,
            previous_words=previous_words,
            CDI_month_dict=CDI_month_dict,
        )

        if metric_results:
            row = [month]  # Start with month
            for metric_name in metric_lst:
                idx = metric_indices.get(metric_name)
                if idx is not None and idx < len(metric_results):
                    row.append(metric_results[idx])
                else:
                    row.append(None)
            row.append(gen["sent_len"].sum())
            scores.append(row)

        # Update CDI dictionary
        if cum_counts is not None:
            CDI_month_dict = word_dict_manager.write_word_dict(cum_counts, CDI_month_dict)

    if not scores:
        return pd.DataFrame()

    score = pd.DataFrame(scores, columns=["month"] + metric_lst + ["word_num"])
    score = score.assign(**info_dict)

    return score


def load_CDI_words(CDI_dir: Path, dataset: str, CDI: bool) -> list:
    """Load CDI words for different datasets."""
    if CDI:
        CDI_frame = pd.read_csv(CDI_dir / f"{dataset}_CDI.csv")
        CDI_words = CDI_frame["word"].tolist()
        return CDI_words, dict.fromkeys(CDI_words, 0)
    return [], {}


def load_word_count_est(word_est_dict: dict, month: int, CDI: bool) -> float:
    """Load word_count_est for each month. Returns 0 if month not in dictionary."""
    # Using dict.get() with default value
    return word_est_dict.get(month, 0) if CDI else 0


def main():
    # Args parser
    args = parse_args()
    # load paths
    gen_dir: Path = settings.PATH.DATA_DIR / args.gen_path
    metric_dir: Path = settings.PATH.DATA_DIR / args.metric_path
    ref_dir: Path = settings.PATH.DATA_DIR / args.ref_path
    CDI_dir: Path = settings.PATH.DATA_DIR / args.CDI_path
    word_est_dir: Path = settings.PATH.DATA_DIR / args.word_est_path
    # load monthly estimation dict
    df_est = pd.read_csv(word_est_dir)
    word_est_dict = dict(zip(df_est["month"], df_est["child_month_est"]))
    print(f"Monthly production estimation loaded {word_est_dict}")
    # set CDI parameters
    CDI = True if "CDI" in args.metric_lst else False
    CDI_month_dict = {"dataset": {"chunk": {"model_type": {"temp": {}}}}}
    score_all = pd.DataFrame()
    # loop over datasets   {dataset{chunk{model{temp:{word_count_dict}}}}}
    for dataset in gen_dir.iterdir():
        # load CDI words
        if dataset.is_dir():
            CDI_words, previous_words = load_CDI_words(CDI_dir, dataset.name, CDI)
            # initialize the CDI words with an ampty dictionary; assign different word dictionary
            word_dict = load_dict(settings.dataset_name_dict[dataset.name])
            for month in tqdm((dataset / f"{args.hour_per_year}_hour_per_year" / args.lang).iterdir()):
                word_count_est = load_word_count_est(word_est_dict, int(month.name), CDI)
                for chunk in month.iterdir():
                    for model in chunk.iterdir():
                        if (model / "gen.csv").exists():  # check whether there exsits the full generation file
                            gen = pd.read_csv(model / "gen.csv")
                            # append info frame
                            info_dict = {
                                "dataset": settings.dataset_name_dict[dataset.name],
                                "month": month.name,
                                "chunk": chunk.name,
                                "model_type": model.name,
                            }
                            # compute the metric and update CDI_month_dict
                            score, CDI_month_dict = append_model_metric(
                                gen,
                                args.temp_lst,
                                info_dict,
                                args.metric_lst,
                                args.threshold,
                                CDI_words,
                                args.chunk_size,
                                word_dict,
                                word_count_est,
                                CDI_month_dict,
                            )
                            score_all = pd.concat([score_all, score])
                            print(f"Finish computing metrics from {model.relative_to(gen_dir)}")

    # compute human production
    ref_data = pd.read_csv(ref_dir)
    CDI_words, previous_words = load_CDI_words(CDI_dir, "CHILDES", CDI)
    # load word_dict
    word_dict = load_dict("CHILDES")
    score_human = append_human_metric(
        ref_data,
        args.metric_lst,
        args.threshold,
        CDI_words,
        args.chunk_size,
        word_dict,
        word_est_dict,
        CDI,
        previous_words,
        CDI_month_dict,
    )
    # Reorder columns based on given list
    score_human = score_human[score_all.columns]

    score_all = pd.concat([score_human, score_all])
    score_all.to_csv(metric_dir / f"metric_{args.hour_per_year}.csv")
    print(f"Saving the metric to {metric_dir}/metric_{args.hour_per_year}.csv")


if __name__ == "__main__":
    main()
