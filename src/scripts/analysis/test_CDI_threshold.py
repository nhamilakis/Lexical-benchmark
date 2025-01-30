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


def append_CDI(ref_data: pd.DataFrame, threshold: int, CDI_words: list,word_est_dict:dict,CDI:bool):

    gen_grouped = ref_data.groupby('month')
    # load the selected CDI words
    calculator = CDICalculator(
        word_list=word_list,
        CDI_words=CDI_words,
        threshold=1,
        word_count_est=1
    )
    selected_words = calculator.select_words()

    scores = []
    for month, gen in gen_grouped:
        sent_lst = gen["text"].tolist()
        word_lst = 
        word_count_est = load_word_count_est(word_est_dict,month,CDI)
        # initialize the CDI calculator class
        calculator = CDICalculator(
            word_list=word_list,
            CDI_words=CDI_words,
            threshold=1,
            word_count_est=1
        )
        metric = Metric(data=sent_lst, temp=month, metric_lst=metric_lst, threshold=threshold,
            CDI_words=CDI_words, word_count_est = word_count_est,chunk_size=chunk_size, word_dict=word_dict)
        row = metric.compute_metrics()
        row.append(gen['sent_len'].sum())
        scores.append([x for x in row if x is not None])

    score = pd.DataFrame(scores, columns=["month", *metric_lst, 'word_num'])
    info_dict = {"dataset": "CHILDES", "chunk": "00", "model_type": "human","temp":"1.0"}
    score = score.assign(**info_dict)
    ordered_cols = [col for col in score.columns if col not in metric_lst] + metric_lst
    score = score[ordered_cols]
    return score


def load_word_count_est(word_est_dict: dict, month: int, CDI: bool) -> float:
    """Load word_count_est for each month. Returns 0 if month not in dictionary."""
    # Using dict.get() with default value
    return word_est_dict.get(month, 0) if CDI else 0


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
