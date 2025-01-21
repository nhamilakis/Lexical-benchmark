"""Tools to merge generations from different models."""

import dataclasses
from pathlib import Path

from lexical_benchmark import settings

# /scratch1/projects/lexical-benchmark/v2/models/ChildRealistic/by_month/EN/12/00/LSTM

class ModelGen:
    """The typing of a row mapping STELA Audio files."""

    


def concat_gen(df:pd.DataFrame,val_lst:list,colname_lst:list)->pd.DataFrame:
    """
    concat gen in differnt files
    final col: month,file_id,text,sent_len,model,temp,gen,estimation,model_type,chunk
    """
    # get the columnds to be splitted and merged
    id_vars, value_vars = split_df_col(df,'model')
    # convert multi-column into one single column
    melted_df = pd.melt(
    df,
    id_vars = id_vars,
    value_vars = value_vars,
    var_name='temp',
    value_name='gen_raw'
    )
    melted_df['temp'] = melted_df['temp'].str.replace('unprompted_', '')
    # append the values to the df
    n = 0
    while n < len(val_lst):
        melted_df[colname_lst[n]] = val_lst[n]
        n += 1

    # Pop the column and add it back
    col_to_move = melted_df.pop('gen_raw')
    melted_df['gen_raw'] = col_to_move

    # apply cleaning on the generation codes
    melted_df['gen'] = melted_df['gen_raw'].apply(char2word)
    return melted_df


def main():
    gen_dir: Path = settings.PATH.DATA_DIR / args.gen_dir
    out_dir: Path = settings.PATH.DATA_DIR / args.out_dir

    # first merge all the files
    gen_all = pd.DataFrame()
    for estimation in tqdm(gen_dir.iterdir()):
        if estimation.is_dir():
            for dataset in estimation.iterdir():
                for month in (dataset/args.estimation/args.lang).iterdir():
                    for chunk in month.iterdir():
                        for model in chunk.iterdir():
                            if (model/args.filename).exists():
                                # load file
                                gen = pd.read_csv(model/args.filename).loc[:, 'month':]
                                val_lst = [estimation.name,model.name,chunk.name]
                                colname_lst = ['estimation','model_type','chunk']
                                converted_gen = concat_gen(gen,val_lst,colname_lst)
                                gen_all = pd.concat([gen_all,converted_gen])
                                print(f'Concatenate file with total row number {converted_gen.shape[0]}')

    #gen_all.to_csv(out_dir/args.filename)        
    # redistribute the results across different months
    
    print(f'Saving the concatenated generation to {out_dir/args.filename}')

