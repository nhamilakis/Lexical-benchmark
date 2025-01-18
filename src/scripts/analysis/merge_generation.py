"""
Merge generations among differnet months
"""
import pandas as pd
from pathlib import Path
from lexical_benchmark import settings
from tqdm import tqdm
import math
from lexical_benchmark.utils.analysis_util import split_df_col

'''
month,file_id,text,sent_len,model,unprompted_0.3,unprompted_0.6,unprompted_1.0,unprompted_1.5
1000/ChildRealistic/by_month/EN/12/00/LSTM
'''
filename = 'gen.csv'
lang = 'EN'
gen_dir: Path = settings.PATH.DATA_DIR / "gen"/"merged"



def concat_gen(df:pd.DataFrame,val_lst:list,colname_lst:list)->pd.DataFrame:
    """
    concat gen in differnt 
    final col: month,file_id,text,sent_len,model,temp,gen,estimation,model_type,chunk
    """

    id_vars, value_vars = split_df_col(df,'model')

    melted_df = pd.melt(
    df,
    id_vars = id_vars,
    value_vars = value_vars,
    var_name='temp',
    value_name='gen'
    )
    melted_df['temp'] = melted_df['temp'].str.replace('unprompted_', '')
    # append the values to the df
    n = 0
    while n < len(val_lst):
        melted_df[colname_lst[n]] = val_lst[n]
        n += 1

    # Pop the column and add it back
    col_to_move = melted_df.pop('gen')
    melted_df['gen'] = col_to_move

    return melted_df

#TODO: apply the cleaning

for estimation in gen_dir.iterdir():
    if estimation.is_dir():
        for dataset in estimation.iterdir():
            for month in (dataset/"by_month"/lang).iterdir():
                for chunk in month.iterdir():
                    for model in chunk.iterdir():
                        if (model/filename).exists():
                            # load file
                            gen = pd.read_csv(model/filename).loc[:, 'month':]
                            val_lst = [estimation.name,model.name,chunk.name]
                            colname_lst = ['estimation','model_type','chunk']
                            converted_gen = concat_gen(gen,val_lst,colname_lst)
                            print(converted_gen.columns)
                            print(converted_gen.head(5))
