# generate the filedir for bash array
import argparse
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from lexical_benchmark import settings
from lexical_benchmark.utils.format_util import *


def parseargs():
    # Run parameters
    parser = argparse.ArgumentParser(description='Get the array script for model training')
    parser.add_argument('--OutPath', type=str, default='.',
                      help='Directory to save path file')
    parser.add_argument('--Resume', default = 'True',
                      help='whether to check there exists the finished job')
    parser.add_argument('--target_model_lst', default = ['LSTM'],
                      help='the target model to be trained; used to check and specify the model dir')
    parser.add_argument('--target_dataset', default = [],
                      help='only load the target dataset; if empty include all')
    parser.add_argument('--target_month', default = [],
                      help='only load the target month for training; if empty include all')
    parser.add_argument('--max_num', default = 2,
                      help='max number of models, if 0 include all')
    parser.add_argument('--train_file', default = 'char_hf.txt',
                      help='name of the train file')
    parser.add_argument('--dev_file', default = 'char_hf.txt',
                      help='name of the dev file')
    parser.add_argument('--lang', default = 'EN',
                      help='language to test')
    return parser.parse_args()




def main():
    # Args parser
    args = parseargs()

    root_dir: Path = settings.PATH.dataset_root
    model_dir: Path = settings.PATH.DATA_DIR / "models"

    data_dirs = []
    model_dirs = []
    dev_dirs = []
    
    
    print('Filter by the given dataset')
    dir_filter = DirectoryFilter(root_dir)
    dataset_dirs = dir_filter.filter_subdirs_by_name(args.target_dataset)

    
    for parent_folder in tqdm(dataset_dirs):
        dev_dir = parent_folder/'dev'/args.lang/args.dev_file
        monthly_path = parent_folder/'by_month'/args.lang
        if not monthly_path.exists():
            print(f"Monthly path does not exist: {monthly_path}")
            continue
        
        # Filter by target month
        print('Filter by the target month')
        model_filter = DirectoryFilter(monthly_path)
        target_month = [str(num) for num in args.target_month]
        month_dirs = model_filter.filter_subdirs_by_name(target_month)
        
        for month_dir in month_dirs:
            # Filter by chunk numbers
            print('Filter by the chunk numbers')
            chunk_filter = DirectoryFilter(month_dir)
            target_month_dirs = chunk_filter.filter_subdirs_by_count(args.max_num)

            # check whether the target model has been trained in the MODEL directory
            for target_month_dir in target_month_dirs:
                target_model_dir = Path(str(target_month_dir).replace('datasets', 'models'))
                
                # check whether the month exists, add them directly to the model_dir
                if not target_model_dir.exists():
                    # loop over the target model
                    for model in args.target_model_lst: 
                        data_dirs.append((target_month_dir/args.train_file).relative_to(root_dir))
                        model_dirs.append((target_model_dir/model).relative_to(model_dir))
                        dev_dirs.append(dev_dir.relative_to(root_dir))

                else:
                    # check whether the model has been trained
                    month_filter = DirectoryFilter(target_model_dir)
                    sub_month_dirs = []
                    for model in args.target_model_lst: 
                        sub_month_dir = month_filter.filter_subdirs_by_name(model)
                        sub_month_dirs.extend(sub_month_dir)
                    
                    for model_path in sub_month_dirs:
                        data_path = Path(str(model_path).replace('models','datasets'))
                        if str_to_bool(args.Resume):
                            if not (model_path/'pytorch_model.bin').exists():
                                data_dirs.append((target_month_dir/args.train_file).relative_to(root_dir))
                                model_dirs.append(model_path.relative_to(model_dir))
                                dev_dirs.append(dev_dir.relative_to(root_dir))
                            else:
                                print(f"The target model already exists: {model_path}")
                        else:
                            print('Ignore the trained model, train from scratch')
                            data_dirs.append((target_month_dir/args.train_file).relative_to(root_dir))
                            model_dirs.append(model_path.relative_to(model_dir))
                            dev_dirs.append(dev_dir.relative_to(root_dir))
    
    # if multiple model, name it in the comprehensive convention; otherwise after the model name
    if len(args.target_model_lst)>1:
        filename = "train-args.index"
    else:
        filename = f"{args.target_model_lst[0]}_train-args.index"

    if data_dirs:  # Only save if we have results
        file_df = pd.DataFrame([data_dirs, dev_dirs,model_dirs]).T
        file_df.to_csv(Path(args.OutPath)/filename, index=False, header=False, sep=" ")
        print(f'Write the result to {args.OutPath}/{filename}')
    else:
        print("No matching directories found based on the given criteria")



if __name__ == "__main__":
    main()
