import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lexical_benchmark import settings
from lexical_benchmark.utils import format_util


def parseargs():
    parser = argparse.ArgumentParser(description="Get the array script for model training")
    parser.add_argument(
        "--OutPath",
        type=str,
        default="/scratch1/projects/lexical-benchmark/v2/datasets/script_arg",
        help="Directory to save path file",
    )
    parser.add_argument(
        "--ModelPath",
        type=str,
        default="models",
        help="relative model path",
    )
    parser.add_argument("--resume", action="store_true", help="Whether to resume from previous ckpt: True or False")
    parser.add_argument(
        "--target_model", default=[], help="the target model to be trained; used to check and specify the model dir"
    )
    parser.add_argument("--target_dataset", default=[], help="only load the target dataset; if empty include all")
    parser.add_argument(
        "--target_month", default=[], help="only load the target month for training; if empty include all"
    )
    parser.add_argument("--max_num", default=2, help="max number of models, if 0 include all")
    parser.add_argument("--train_file", default="char_hf.txt", help="name of the train file")
    parser.add_argument("--dev_file", default="char_hf.txt", help="name of the dev file")
    parser.add_argument("--lang", default="EN", help="language to test")
    return parser.parse_args()


def process_target_model_dir(target_month_dir, target_model_dir, args, root_dir, model_dir, dev_dir):
    """Process a target model directory and return data, model, and dev directories."""
    datasets, splits, chunks,dev_dirs = [], [], [], []

    if not target_model_dir.exists():
        # Add new model paths
        for model in args.target_model_lst:
            datasets.append(target_model_dir.name)
            splits.append(target_model_dir.parent)
            chunks.append(target_model_dir.parent[-1])
            dev_dirs.append(dev_dir.relative_to(root_dir))
    else:
        # Handle existing model directories
        month_filter = format_util.DirectoryFilter(target_model_dir)
        sub_month_dirs = []
        for model in args.target_model_lst:
            sub_month_dirs.extend(month_filter.filter_subdirs_by_name(model))

        for model_path in sub_month_dirs:
            if args.resume:
                if not (model_path / "training_args.bin").exists():
                    datasets.append(target_model_dir.name)
                    splits.append(target_model_dir.parent)
                    chunks.append(target_model_dir.parent[-1])
                    dev_dirs.append(dev_dir.relative_to(root_dir))
                else:
                    print(f"The target model already exists: {model_path}")
            else:
                print("Ignore the trained model, train from scratch")
                datasets.append(target_model_dir.name)
                splits.append(target_model_dir.parent)
                chunks.append(target_model_dir.parent[-1])
                dev_dirs.append(dev_dir.relative_to(root_dir))

    return datasets, splits, chunks,dev_dirs


def main():
    args = parseargs()
    root_dir: Path = settings.PATH.dataset_root
    model_dir: Path = settings.PATH.DATA_DIR / args.ModelPath

    datasets, splits, chunks,dev_dirs  = [], [], [], []

    # Filter datasets
    for dataset 
    # check model directoyr; whether there exists the model 

    # Only save if we have results
    filename = "train-args.index"
    if datasets:
        file_df = pd.DataFrame([datasets, splits, chunks,dev_dirs]).T
        file_df.to_csv(Path(args.OutPath) / filename, index=False, header=False, sep=" ")
        print(f"Write the result to {args.OutPath}/{filename}")
    else:
        print("No matching directories found based on the given criteria")


if __name__ == "__main__":
    main()
