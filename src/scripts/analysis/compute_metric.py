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
    parser.add_argument(
        "--metric_lst",
        type=list,
        default=["type_token_ratio","rej_type_rate","CDI"],
        help="metric list; ttr,rej_type_rate,CDI"
    )
    parser.add_argument("--temp_lst", type=list, default=[0.3, 0.6, 1.0, 1.5], help="temperature list")
    parser.add_argument("--hour_per_year", default=1000, type=int, help="Estimated yearly exposure hours")
    parser.add_argument("--chunk_size", default=500, type=int, help="Chunk size to normalize the scores")
    parser.add_argument("--threshold", default=60, type=int, help="threshold to compute CDI scores")
    parser.add_argument("--lang", default="EN", type=str, help="tested language")
    return parser.parse_args()



def append_model_metric(gen: pd.DataFrame, temp_lst: list, info_dict: dict, metric_lst: list, threshold: int, 
            CDI_words: list, chunk_size: int, word_dict: dict,word_count_est:int):
   scores = []
   for temp in temp_lst:
       sent_lst = gen[f"unprompted_{temp}"].apply(char2word).tolist()
       metric = Metric(data=sent_lst, temp=temp, metric_lst=metric_lst, threshold=threshold, 
                      CDI_words=CDI_words, chunk_size=chunk_size, word_dict=word_dict,word_count_est=word_count_est)
       scores.append([x for x in metric.compute_metrics() if x is not None])

   score = pd.DataFrame(scores, columns=["temp", *metric_lst])
   score = score.assign(**info_dict)
   ordered_cols = [col for col in score.columns if col not in metric_lst] + metric_lst
   score = score[ordered_cols]
   score['word_num'] = gen['sent_len'].sum()
   return score


def append_human_metric(ref_data: pd.DataFrame, metric_lst: list, threshold: int, CDI_words: list, chunk_size: int, 
            word_dict: dict,word_est_dict:dict):
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
   info_dict = {"dataset": "CHILDES", "chunk": "00", "model_type": "human","temp":"1.0"}
   score = score.assign(**info_dict)
   ordered_cols = [col for col in score.columns if col not in metric_lst] + metric_lst
   score = score[ordered_cols]
   return score


def load_CDI_words(CDI_dir:Path,dataset:str,CDI:bool)->list:
    """Load CDI words for different datasets."""
    if CDI:
        CDI_frame = pd.read_csv(CDI_dir/f'{dataset}_CDI.csv')
        return CDI_frame['word'].tolist()
    return []



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

    # set CDI parameters
    CDI=True if "CDI" in args.metric_lst else False

    score_all = pd.DataFrame()
    # loop over datasets
    for dataset in gen_dir.iterdir():
        # load CDI words
        CDI_words = load_CDI_words(CDI_dir,dataset.name,CDI)
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

    # compute human production
    ref_data = pd.read_csv(ref_dir)
    CDI_words = load_CDI_words(CDI_dir,"CHILDES")
    # load word_dict
    word_dict = load_dict("CHILDES")
    score_human = append_human_metric(ref_data,args.metric_lst,args.threshold,CDI_words,args.chunk_size, word_dict)
    # Reorder columns based on given list
    score_human = score_human[score_all.columns]
    score_all = pd.concat([score_human,score_all])
    score_all.to_csv(metric_dir/f"metric_{args.hour_per_year}.csv")
    print(f"Saving the metric to {metric_dir}/metric_{args.hour_per_year}.csv")

if __name__ == "__main__":
    main()
