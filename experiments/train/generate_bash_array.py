# generate the filedir for bash array
import argparse
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from lexical_benchmark import settings
from lexical_benchmark.utils.format_util import str_to_bool

def parseargs():
    # Run parameters
    parser = argparse.ArgumentParser(description='Get the array script for training')
    parser.add_argument('--TrainPath', type=str, default='/scratch1/projects/lexical-benchmark/v2/datasets',
                        help='Path prefix to the train file')
    parser.add_argument('--ModelPath', type=str, default='/scratch1/projects/lexical-benchmark/v2/models',
                      help='Directory to save model checkpoints')
    parser.add_argument('--OutPath', type=str, default='/scratch2/jliu/Lexical-benchmark/experiments/train/hf',
                      help='Directory to save path file')
    parser.add_argument('--Resume', default = 'False',
                      help='whether to check there exists the finished job')
    parser.add_argument('--target_model', default = 'trans',
                      help='the target model to be trained; used to check and specify the model dir')
    parser.add_argument('--Target_dataset', default = [],
                      help='only load the target dataset; if empty include all')
    parser.add_argument('--Target_month', default = [],
                      help='only load the target month for training; if empty include all')
    parser.add_argument('--max_num', default = 2,
                      help='max number of models, if 0 include all')
    parser.add_argument('--lang', default = 'EN',
                      help='language to test')
    return parser.parse_args()




def get_subdirs(path):
    """Get immediate subdirectories from a path."""
    return [d for d in path.iterdir() if d.is_dir()]



def main(argv):

    # Args parser
    args = parseargs()
    root_dir: Path = settings.PATH.dataset_root

    
    print('##########################')
    print('Filter the directory list!')
    print('##########################')

    parent_dirs = list(root_dir.iterdir())

    # Filter by Target_dataset if specified
    if len(args.Target_dataset) > 0:
        parent_dirs = [d for d in parent_dirs if d.name in args.Target_dataset]
    # Filter by Target_month if specified
    if len(args.Target_month) > 0:
        parent_dirs = [d for d in parent_dirs if d.name in args.Target_month]

    print('##########################')
    print('Iterate over the directory')
    print('##########################')

    # Get the subdirectories for each parent
    data_dirs = []
    model_dirs = []
    for parent_folder in tqdm(parent_dirs):
        monthly_path = parent_folder/'by_month'/args.lang
        if monthly_path.exists():
            # Get all month directories
            month_dirs = get_subdirs(monthly_path)
            
            # Process each month directory
            for month_dir in month_dirs:
                # Get subdirectories under each month
                sub_month_dirs = get_subdirs(month_dir)
                if args.max_num > 0:
                    # Sort subdirectories numerically and select first n
                    try:
                        sorted_sub_dirs = sorted(sub_month_dirs, key=lambda x: int(x.name))
                        sub_month_dirs = sorted_sub_dirs[:args.max_num]
                    except ValueError:
                        print(f"Warning: Some subdirectory names in {month_dir} are not numeric. Using default sorting.")
                        sub_month_dirs = sub_month_dirs[:args.max_num]
                
                # Process each path and check if transformed path exists
                for original_path in sub_month_dirs:
                    # Create the transformed path
                    transformed_path = Path(str(original_path).replace('datasets', 'models'))
                    if str_to_bool(args.Resume):
                        # only check the existing checkpoint when setting "Resume" as True
                        print('Checking whether the model has been trained')
                        if not (transformed_path/args.target_model/'pytorch_model.bin').exists():
                            # Only add paths that don't have corresponding transformed versions
                            data_dirs.append(original_path)
                            model_dirs.append(transformed_path/args.target_model)
                        else:
                            print(f"The target model already exists: {transformed_path}")
                    else:
                        print('Ignore the trained model, train from scratch')
                        data_dirs.append(original_path)
                        model_dirs.append(transformed_path/args.target_model)

    
    file_df = pd.DataFrame([data_dirs,model_dirs]).T
    file_df.to_csv(f'{args.OutPath}/{args.target_model}.train',index = False, header = False)
    print(f'Write the result to {args.OutPath}/{args.target_model}.train')




if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
