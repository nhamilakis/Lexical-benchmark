"""Compute core metrics from the generation directory"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lexical_benchmark import settings
from lexical_benchmark.datasets.utils.text_cleaning import char2word
from lexical_benchmark.stats.metric import Metric, load_dict


def parse_args():
    # Run parameters
    parser = argparse.ArgumentParser(description="Finetune decoder-onlyls models")

    parser.add_argument(
        "--gen_path",
        type=str,
        default="gen/merged",
        help="Path to the generated texts",
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        default="gen/merged/CHILDES_model.csv",
        help="Path to the human reference data",
    )
    parser.add_argument(
        "--metric_path",
        type=str,
        default="gen/metrics.csv",
        help="Path to save metrics",
    )
    parser.add_argument(
        "--metric_lst",
        type=list,
        default=["type_token_ratio","rej_type_rate"],
        help="metric list; ttr,rej_type_rate"
    )

    parser.add_argument("--temp_lst", type=list, default=[0.3, 0.6, 1.0, 1.5], help="temperature list")
    parser.add_argument("--hour_per_year", default=1000, type=int, help="Estimated yearly exposure hours")
    parser.add_argument("--chunk_size", default=1000, type=int, help="Chunk size to normalize the scores")
    parser.add_argument("--threshold", default=60, type=int, help="threshold to compute CDI scores")
    parser.add_argument("--lang", default="EN", type=str, help="tested language")
    return parser.parse_args()



def append_model_metric(gen: pd.DataFrame, temp_lst: list, info_dict: dict, metric_lst: list, threshold: int, CDI_words: list, chunk_size: int, word_dict: dict):
   scores = []
   for temp in temp_lst:
       sent_lst = gen[f"unprompted_{temp}"].apply(char2word).tolist()
       metric = Metric(data=sent_lst, temp=temp, metric_lst=metric_lst, threshold=threshold, 
                      CDI_words=CDI_words, chunk_size=chunk_size, word_dict=word_dict)
       scores.append([x for x in metric.compute_metrics() if x is not None])

   score = pd.DataFrame(scores, columns=["temp", *metric_lst])
   score = score.assign(**info_dict)
   ordered_cols = [col for col in score.columns if col not in metric_lst] + metric_lst
   score = score[ordered_cols]
   score['word_num'] = gen['sent_len'].sum()
   return score

def append_human_metric(ref_data: pd.DataFrame, metric_lst: list, threshold: int, CDI_words: list, chunk_size: int, word_dict: dict):
   gen_grouped = ref_data.groupby('month')
   scores = []
   for month, gen in gen_grouped:
       sent_lst = gen["text"].tolist()
       metric = Metric(data=sent_lst, temp=month, metric_lst=metric_lst, threshold=threshold,
                      CDI_words=CDI_words, chunk_size=chunk_size, word_dict=word_dict)
       row = metric.compute_metrics()
       row.append(gen['sent_len'].sum())
       scores.append([x for x in row if x is not None])

   score = pd.DataFrame(scores, columns=["month", *metric_lst, 'word_num'])
   info_dict = {"dataset": "CHILDES", "chunk": "00", "model_type": "human"}
   score = score.assign(**info_dict)
   ordered_cols = [col for col in score.columns if col not in metric_lst] + metric_lst
   score = score[ordered_cols]
   return score


def main():
    # Args parser
    args = parse_args()
    gen_dir: Path = settings.PATH.DATA_DIR / args.gen_path
    metric_dir: Path = settings.PATH.DATA_DIR / args.metric_path
    ref_dir: Path = settings.PATH.DATA_DIR / args.ref_path

    score_all = pd.DataFrame()
    CDI_words = []

    # loop over datasets
    for dataset in gen_dir.iterdir():
        if dataset.is_dir():
            # assign different word dictionary
            word_dict = load_dict(settings.dataset_name_dict[dataset.name])

            for month in tqdm((dataset / f"{args.hour_per_year}_hour_per_year" / args.lang).iterdir()):
                for chunk in month.iterdir():
                    for model in chunk.iterdir():
                        if (model/"gen.csv").exists():   # check whether there exsits the full generation file
                            gen = pd.read_csv(model/"gen.csv")
                            # append info frame
                            info_dict = {
                                    "dataset": settings.dataset_name_dict[dataset.name],
                                    "month": month.name,
                                    "chunk": chunk.name,
                                    "model_type": model.name
                                }
                            # compute the metric here sent_lst:list,temp_lst:list,info_dict:dict,
                            score = append_model_metric(gen,args.temp_lst,info_dict,args.metric_lst,
                                                args.threshold,CDI_words,args.chunk_size, word_dict)
                            score_all = pd.concat([score_all, score])
                            print(f"Finish computing metrics from {model.relative_to(gen_dir)}")

    score_all.to_csv(metric_dir)
    print(f"Saving the metric to {metric_dir}")

    # compute human production
    ref_data = pd.read_csv(ref_dir)
    # load word_dict
    word_dict = load_dict("CHILDES")
    score_human = append_human_metric(ref_data,args.metric_lst,args.threshold,CDI_words,args.chunk_size, word_dict)
    score_human.to_csv(metric_dir.parent/"human_metric.csv")
    print(f"Saving the metric to {metric_dir.parent}/human_metric.csv")


if __name__ == "__main__":
    main()
