import pandas as pd
from pathlib import Path
import math
from lm_benchmark.plot_util import *
from lm_benchmark.utils import *
from lm_benchmark.settings import ROOT
import argparse
import sys

# set constant setting
freq_ROOT = f'{ROOT}/datasets/processed/month_count/'
gen_ROOT = f'{ROOT}/datasets/processed/generation/'
FIG_SIZE = (10, 10)
month_range = [6,36]

def arguments() -> argparse.Namespace:
    """Build & Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", default=f"train")
    parser.add_argument("--MODEL", default=f"CHILDES")
    parser.add_argument("--PROMPT", default=f"prompted")
    return parser.parse_args()


def select_rows(candi, target):
    """
    select rows based on the target sum;
    return the rest of the dataset and the selected rows
    """
    cumulative_sum = candi['num_tokens'].cumsum()
    selected_rows = candi[cumulative_sum <= target]
    rest_df = candi[cumulative_sum > target]
    return selected_rows, rest_df


def chunk_dataframe(df, column_name, trans_header, m, n, out_path):
    # Ensure the output path exists
    Path(out_path).mkdir(parents=True, exist_ok=True)
    tc_lst = []
    sub_dataframes = []

    chunk_number = 1

    while chunk_number <= n:
        # select rows based on the target sum
        current_chunk_df, df = select_rows(df, m)
        count_df = TokenCount.from_df(current_chunk_df, trans_header)
        tc_lst.append(count_df)
        # Save to CSV
        current_chunk_df.to_csv(Path(out_path) / f'{chunk_number}.csv', index=False)
        chunk_number += 1

    # Get the summary statistics
    child_stats = tc_summary(tc_lst[:n])
    mean_nonword_type = child_stats['p_nonword_type'].mean()
    mean_nonword_token = child_stats['p_nonword_token'].mean()

    return sub_dataframes[:n], mean_nonword_type, mean_nonword_token


def custom_round(value):
    return math.floor(value)


def main() -> None:
    args = arguments()
    MODEL = args.MODEL
    PROMPT = args.PROMPT
    header = args.header
    target = f'{PROMPT}_{MODEL}.csv'
    gen_path = gen_ROOT
    if MODEL == 'CHILDES':
        out_ROOT = Path(freq_ROOT) / 'non_cum_norm' / MODEL
        trans_path = Path(gen_path) / 'unprompted_LSTM.csv'

    else:
        out_ROOT = Path(freq_ROOT)/'non_cum_norm'/PROMPT/MODEL/header
        trans_path = Path(gen_path)/Path(target)

    trans = pd.read_csv(trans_path)

    vocal = pd.read_csv(ROOT + '/datasets/raw/vocal_month.csv')
    sel = vocal[(vocal['month']>=6) & (vocal['month']<=36)][['child_num_tokens','month']]
    # get the sum of first few months
    sel = sum_rows(sel, 9,'child_num_tokens')
    target_month = 16

    chunk_size = sel.at[target_month-1, 'child_num_tokens']
    # get chunk number
    sel['chunk_num'] = sel['child_num_tokens']/chunk_size
    sel['chunk_num_round'] = sel['chunk_num'].apply(custom_round)

    # gather the first few months


    stat_frame = pd.DataFrame()
    # segment into different chunks
    n = 0
    while n < sel.shape[0]:

        month = sel['month'].tolist()[n]
        out_path = out_ROOT / str(month)
        # get non-word rate of each chunk

        if n == 0:     # gather the first few months
            print(f'loading files from the first {month} months')
            month_trans = trans[trans['month'] <= month]
        if n > 0:     # gather by months
            print(f'loading files in month {month}')
            # get non-word rate of each chunk
            month_trans = trans[trans['month']==month]

        chunk_num = sel['chunk_num_round'].tolist()[n]
        # segment into chunks
        try:
            subdf,nonword_type,nonword_token = chunk_dataframe(month_trans,'num_tokens',header,chunk_size,chunk_num,out_path)
            print(f'Finished segmenting files in month {month}')
            if n == 0:
                month = math.floor((target_month-month)/2)
            stat_temp = pd.DataFrame([[month,nonword_type,nonword_token]])
            stat_frame = pd.concat([stat_frame,stat_temp])
        except:
            print(month)
        n += 1

    stat_frame.columns = ['month','p_nonword_type','p_nonword_token']
    stat_frame.to_csv((out_ROOT/'stat.csv'))



if __name__ == "__main__":
    args = sys.argv[1:]
    main()