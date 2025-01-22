# generate the filedir for bash array
import argparse
import sys
from pathlib import Path

import pandas as pd
from lexical_benchmark import settings
from lexical_benchmark.utils import format_util
from tqdm import tqdm


def parseargs():
    # Run parameters
    parser = argparse.ArgumentParser(description="Get the array script for generation")
    parser.add_argument("--GenPath", type=str, default="gen/merged", help="Generation root directory")
    parser.add_argument("--OutPath", type=str, default="datasets/script_arg/generation-args.index", help="Directory to save path file")
    parser.add_argument("--Resume", default="True", help="whether to check there exists the finished job")
    parser.add_argument(
        "--target_model", default="", help="the target model to be trained; used to check and specify the model dir"
    )
    parser.add_argument(
        "--target_dataset", default=[], help="only load the target dataset; if empty include all"
    )
    parser.add_argument(
        "--target_month",
        default=[],
        help="only load the target month for training; if empty include all",
    )
    parser.add_argument("--max_num", default=0, help="max number of models, if 0 include all")
    parser.add_argument("--lang", default="EN", help="language to test")
    return parser.parse_args()


def main() -> None:
    """Build Generation."""
    # Args parser
    args = parseargs()
    root_dir: Path = settings.PATH.DATA_DIR / "models"
    gen_root = settings.PATH.DATA_DIR / args.GenPath
    OutPath = settings.PATH.DATA_DIR / args.OutPath

    data_dirs = []
    model_dirs = []

    # Filter by dataset
    print("Filter by the given dataset")
    dir_filter = format_util.DirectoryFilter(root_dir)
    dataset_dirs = dir_filter.filter_subdirs_by_name(args.target_dataset)

    for parent_folder in tqdm(dataset_dirs):
        monthly_path = parent_folder / "by_month" / args.lang
        if not monthly_path.exists():
            print(f"Monthly path does not exist: {monthly_path}")
            continue

        # Filter by model type
        print("Filter by the target month")
        model_filter = format_util.DirectoryFilter(monthly_path)
        target_month = [str(num) for num in args.target_month]
        month_dirs = model_filter.filter_subdirs_by_name(target_month)
        print(month_dirs)
        for month_dir in month_dirs:
            # Filter by chunk numbers
            print("Filter by the chunk numbers")
            chunk_filter = format_util.DirectoryFilter(month_dir)
            target_month_dirs = chunk_filter.filter_subdirs_by_count(args.max_num)

            for target_month_dir in target_month_dirs:
                # Filter by target month
                print("Filter by the model type")
                month_filter = format_util.DirectoryFilter(target_month_dir)
                sub_month_dirs = month_filter.filter_subdirs_by_name(args.target_model)
                # Process paths
                for original_path in sub_month_dirs:
                    # Check model training completion
                    if not (original_path / "training_args.bin").exists():
                        print(f"Skip due to untrained model: {original_path}")
                        continue
                    transformed_path = Path(str(original_path).replace("models", "gen/merged"))
                    if format_util.str_to_bool(args.Resume):
                        if not (transformed_path / "gen.csv").exists():
                            data_dirs.append(original_path.relative_to(root_dir))
                            model_dirs.append(transformed_path.relative_to(gen_root))
                        else:
                            print(f"The target generation already exists: {transformed_path}")
                    else:
                        print("Ignore the finished generation, generate from scratch")
                        data_dirs.append(original_path.relative_to(root_dir))
                        model_dirs.append(transformed_path.relative_to(gen_root))
    if data_dirs:  # Only save if we have results
        file_df = pd.DataFrame([data_dirs, model_dirs]).T

        file_df.to_csv(OutPathnano, index=False, header=False, sep=" ")
        print(f"Write the result to {OutPath}")
    else:
        print("No matching directories found based on the given criteria")


if __name__ == "__main__":
    main()
