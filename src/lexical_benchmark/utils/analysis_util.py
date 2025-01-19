import pandas as pd


def split_df_col(df:pd.DataFrame,target_col:str)->list[str]:
    # Get column names
    cols = df.columns.tolist()

    # Find indices of target columns
    target_cols = [target_col]
    target_indices = [cols.index(col) for col in target_cols]

    # Get columns to the left of first target
    left_cols = cols[:min(target_indices)+1]
    # Get columns to the right of last target
    right_cols = cols[max(target_indices) + 1:]
    return left_cols, right_cols



