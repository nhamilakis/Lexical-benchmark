"""Compute core metrics from the generation directory"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lexical_benchmark import settings
from lexical_benchmark.datasets import childes
from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.datasets.utils.text_cleaning import char2word
from lexical_benchmark.stats.metric import Metric


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



def load_dict(dataset_name:str):
    """Load dictionary based on different datasets"""
    if dataset_name=='child':
        print("Append en dict with adult input")
        dataset = childes.CHILDESDataset()
        childes_adult_extras_lexique = childes.CHILDESExtrasLexicon(dataset)
        childes_adult_extras_lexique.add_lang("Eng-NA", "adult")
        childes_adult_extras_lexique.add_lang("Eng-UK", "adult")
        dict_hash_id = childes_adult_extras_lexique.cache_current()
        en_dict = dataset_utils.DictionairyCleaner(lang="EN", childes_extra_id=dict_hash_id)
    else:
        en_dict = dataset_utils.DictionairyCleaner(lang="EN")
    print("Dictionary has been loaded!")
    return en_dict



def compute_metric(sent_lst:list,temp:str,metric_lst:list,threshold:int,CDI_words:list,chunk_size:int,word_dict:dict)->list:
    """Compute metric scores for a list of texts."""
    data = [word for sentence in sent_lst for word in sentence.split()]
    metric = Metric(data=data, chunk_size=chunk_size, word_dict=word_dict)
    row = [temp]
    row.extend([
           metric.compute_ttr() if "type_token_ratio" in metric_lst else None,
           metric.compute_type_rej_rate() if "rej_type_rate" in metric_lst else None,
           metric.compute_CDI(threshold, CDI_words) if "CDI" in metric_lst else None
       ])
    return row



def append_model_metric(gen: pd.DataFrame,temp_lst:list,info_dict:dict,metric_lst:list,threshold:int,CDI_words:list,chunk_size:int, word_dict:dict):
    """Compute metric scores for generated text."""
    scores = []
    for temp in temp_lst:
        sent_lst = gen[f"unprompted_{temp}"].apply(char2word).tolist()
        row = compute_metric(sent_lst,temp,metric_lst,threshold,CDI_words,chunk_size,word_dict)
        scores.append([x for x in row if x is not None])
    # append info frame
    score = pd.DataFrame(scores, columns=["temp", *metric_lst])
    score = score.assign(**info_dict)
    # move the scores on the right
    ordered_cols = [col for col in score.columns if col not in metric_lst] + metric_lst
    score = score[ordered_cols]
    return score



def main():
    # Args parser
    args = parse_args()
    gen_dir: Path = settings.PATH.DATA_DIR / args.gen_path
    metric_dir: Path = settings.PATH.DATA_DIR / args.metric_path

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
                            score = append_model_metric(gen,args.temp_lst,info_dict,args.metric_lst,args.
                                                    threshold,CDI_words,args.chunk_size, word_dict)
                            score_all = pd.concat([score_all, score])
                            print(f"Finish computing metrics from {model.relative_to(gen_dir)}")


    score_all.to_csv(metric_dir)
    print(f"Saving the metric to {metric_dir}")


if __name__ == "__main__":
    main()
