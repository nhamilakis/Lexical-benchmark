"""Merge generations among differnet months"""
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from lexical_benchmark import settings
from lexical_benchmark.utils.analysis_util import split_df_col
from lexical_benchmark.datasets.utils.text_cleaning import char2word



def parse_args():
    # Run parameters
    parser = argparse.ArgumentParser(description="Concartenate generated sequences")

    parser.add_argument(
        "--gen_dir",
        type=str,
        default="gen/merged",
        help="realtive path to the root dir",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="gen/merged",
        help="realtive path to save the concatenated generation",
    )

    parser.add_argument("--filename", type=str, default="gen.csv", help="gen file name")
    parser.add_argument("--lang", type=str, default='EN', help="random seed")
    return parser.parse_args()


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
    # Args parser
    args = parse_args()
    gen_dir: Path = settings.PATH.DATA_DIR / args.gen_dir
    out_dir: Path = settings.PATH.DATA_DIR / args.out_dir

    '''
    gen_all = pd.DataFrame()
    for estimation in tqdm(gen_dir.iterdir()):
        if estimation.is_dir():
            for dataset in estimation.iterdir():
                for month in (dataset/args.lang).iterdir():
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

    gen_all.to_csv(out_dir/args.filename)                         
    print(f'Saving the concatenated generation to {out_dir/args.filename}')
    '''
    # save the file recursively; 
    # /scratch1/projects/lexical-benchmark/v2/models/ChildRealistic/by_month/EN/12/00/LSTM
    gen_all = pd.read_csv(out_dir/args.filename)
    col_lst = ['estimation','month','chunk','model_type']
    gen_grouped = gen_all.groupby(col_lst)
    for group, gen_group in gen_grouped:
        # only select partial 
        gen_group = gen_group[]
        # save the generation to the target file
        file_dir = gen_dir/'1000'/group[0]/f"{group[1]:02d}"/f"{group[2]:02d}"/group[3]
        file_dir.mkdir(parents=True, exist_ok=True)
        gen_group.to_csv(file_dir/'gen.csv')
        print(f'Saving the result to {file_dir}')


        

if __name__ == "__main__":
    main()
