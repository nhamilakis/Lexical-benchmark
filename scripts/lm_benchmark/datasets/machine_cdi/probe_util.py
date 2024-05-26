"""util functions to create probe sets"""
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from scipy.stats import ttest_rel
from lm_benchmark.utils import TokenCount
from lm_benchmark.plot_util import tc_compute_miss_oov_rates
def sort_files(directory:Path,suffix:str)->dict:
    # Get list of files in the directory
    files = [file.name for file in directory.iterdir() if file.name.endswith(suffix)]
    # Extract epoch and batch numbers from file names
    file_info = [(file, int(file.split('_')[0]), int(file.split('_')[1].split('.')[0])) for file in files]
    # Sort files based on epoch and then batch numbers
    sorted_files = sorted(file_info, key=lambda x: (x[1], x[2]))
    # Get sorted file names
    sorted_file_names = [file[0] for file in sorted_files]
    return sorted_file_names


def load_files(file_names: list, parent_path: Path)->TokenCount:
    """
    Load files as TC objects

    Args:
    - file_names (list): List of filenames in the desired order.
    - parent_dir (str): Path to the common parent directory.

    Returns:
    - dataframes (list): List of TokwnCounts read from the files.
    """
    dataframes = {}

    for file_name in file_names:
        file_path = parent_path / file_name
        if file_path.name.endswith('txt'):
            # load as TC object
            word_count = TokenCount.from_text_file(file_path)
        elif file_path.name.endswith('csv'):
            word_count = TokenCount.from_df(file_path, 'LSTM_segmented')

        word_count.name = file_name[:-4]
        dataframes[file_name[:-4]] = word_count

    return dataframes


def select_probe_set(files: dict, out_dir: Path) -> dict:
    """
    input: a dictionary of sorted files
    return: a dictionary of unique probe files
    """
    # Create out_dir if it does not exist
    out_dir.mkdir(parents=True, exist_ok=True)
    # Iterate over batches (T-1, T, T+1) recursively
    stat_lst = []
    dataframes = {}
    file_keys = list(files.keys())
    for i in tqdm(range(1, len(file_keys) - 1)):
        # Load word count CSV files for batches T-1, T, T+1
        prev_file = file_keys[i - 1]
        curr_file = file_keys[i]
        next_file = file_keys[i + 1]
        df_prev = files[prev_file].df
        df_curr = files[curr_file].df
        df_next = files[next_file].df

        # Get the words from the current batch
        selected_words = df_curr['word'].tolist()
        # Remove words overlapping with batches T-1 and T+1
        selected_words = [word for word in selected_words if word not in df_prev['word'].tolist() and word not in df_next['word'].tolist()]
        selected_df = df_curr[df_curr['word'].isin(selected_words)]

        # Get statistics for the CSV file
        stat_lst.append([curr_file, selected_df.shape[0]])

        # Convert it to TokenCount object
        word_count = TokenCount()
        word_count.name = curr_file
        word_count.df = selected_df
        dataframes[curr_file] = word_count

        # Write probe set to file
        selected_df.to_csv(out_dir / f"batch_{i}.csv", index=False)
        print(f"Probe set for batch {curr_file} saved to {out_dir}")

    stat_df = pd.DataFrame(stat_lst, columns=['filename', 'token_count'])
    stat_df.to_csv(out_dir / 'stat_probe.csv', index=False)
    return dataframes

def compare_scores(probe_files:list,gen_files:list)-> dict:
    stat = pd.DataFrame()
    for filename,probe_tc in probe_files.items():
        # select from the corresponding results
        msc_prev, osc_prev, nsc_prev = tc_compute_miss_oov_rates(probe_tc,gen_files[filename.split('_')[0]+'_'+str(int(filename.split('_')[1])-1)],groupbin=1)
        msc_cur, osc_cur, nsc_cur = tc_compute_miss_oov_rates(probe_tc,gen_files[filename],groupbin=1)
        msc_next, osc_next, nsc_next = tc_compute_miss_oov_rates(probe_tc,gen_files[filename.split('_')[0]+'_'+str(int(filename.split('_')[1])+1)],groupbin=1)
        stat_temp = pd.DataFrame([[msc_prev['dfreq_score'].tolist()[0],msc_prev['pmiss'].tolist()[0], osc_prev['poov'].tolist()[0], nsc_prev['pnword'].tolist()[0],
                                   msc_cur['dfreq_score'].tolist()[0], msc_cur['pmiss'].tolist()[0], osc_cur['poov'].tolist()[0], nsc_cur['pnword'].tolist()[0],
                                   msc_next['dfreq_score'].tolist()[0], msc_next['pmiss'].tolist()[0], osc_next['poov'].tolist()[0], nsc_next['pnword'].tolist()[0]]])
        stat = pd.concat([stat,stat_temp])
    stat.columns = ['freq_score_prev','pmiss_prev','poov_prev','pnword_prev',
                    'freq_score_cur','pmiss_cur','poov_cur','pnword_cur',
                    'freq_score_next','pmiss_next','poov_next','pnword_next']

    # run stat analysis
    result = {}
    score_lst = ['freq_score', 'pmiss', 'poov', 'pnword']
    for score in score_lst:
        t_learn, p_learn = ttest_rel(stat[f'{score}_cur'], stat[f'{score}_prev'])
        t_forget, p_forget = ttest_rel(stat[f'{score}_cur'], stat[f'{score}_next'])
        result[f'{score}_learn'] = p_learn
        result[f'{score}_forget'] = p_forget

    return result

