"""util functions to create probe sets"""
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from lm_benchmark.utils import TokenCount

def sort_files(directory:Path,suffix:str)->list:
    # Get list of files in the directory
    files = [file.name for file in directory.iterdir() if file.name.endswith('txt')]
    # Extract epoch and batch numbers from file names
    file_info = [(file, int(file.split('_')[0]), int(file.split('_')[1].split('.')[0])) for file in files]
    # Sort files based on epoch and then batch numbers
    sorted_files = sorted(file_info, key=lambda x: (x[1], x[2]))
    # Get sorted file names
    sorted_file_names = [file[0] for file in sorted_files]
    return sorted_file_names


def load_files(file_names: list, parent_path: Path) -> list:
    """
    Load files as TC objects

    Args:
    - file_names (list): List of filenames in the desired order.
    - parent_dir (str): Path to the common parent directory.

    Returns:
    - dataframes (list): List of DataFrames read from the files.
    """
    dataframes = []

    for file_name in file_names:
        file_path = parent_path / file_name
        if file_path.is_file():
            # load as TC object
            word_count = TokenCount.from_text_file(file_path)
            df = word_count.df
            dataframes.append(df)
        else:
            print(f"File not found: {file_path}")

    return dataframes


def select_probe_set(files:list,out_dir:Path)->pd.DataFrame:
    """
    input: a list of sorted files
    return: a list of unique probe files
    """
    # create out_path dir if there is not
    out_dir.mkdir(parents=True, exist_ok=True)
    # Iterate over batches (T-1, T, T+1) recursively
    stat_lst = []
    for i in tqdm(range(1, len(files) - 1)):
        # Load word count CSV files for batches T-1, T, T+1
        dfs = files[i - 1: i + 2]
        '''
        # remove 25% of words with lowest frequency in batch T
        num_selected_words = max(int(len(dfs[1]) * 0.75), 1)
        df_t_sorted = dfs[1].sort_values(by='count', ascending=True)
        selected_words = df_t_sorted.head(num_selected_words)['word'].tolist()
        '''
        selected_words = dfs[1]['word'].tolist()
        # Remove words overlapping with batches T-1 and T+1
        selected_words = [word for word in selected_words if word not in dfs[0]['word'] and word not in dfs[1]['word']]
        selected_df = dfs[1][dfs[1]['word'].isin(selected_words)]
        # get stat for the .csv file
        stat_lst.append([files[i],selected_df.shape[0]])
        # Write probe set to file
        selected_df.to_csv(out_dir / f"batch_{i}.csv", index=False)
        print(f"Probe set for batch {files[i]} saved to {out_dir}")

    stat_df = pd.DataFrame(stat_lst, columns=['filename', 'token_type'])
    stat_df.to_csv(out_dir / 'stat_probe.csv', index=False)

    return stat_df




