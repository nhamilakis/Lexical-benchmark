#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from lm_benchmark.utils import TokenCount

def merge_df(merged_df, df2, header: str,month:int):    # TODO: preserve the month group info
    # Merge freq dataframes on 'word' column
    count_df = TokenCount.from_df(df2, header)
    df2 = count_df.df[['word', 'freq_m']]
    # Merge freq dataframes on index 'word'
    merged_df = pd.merge(merged_df, df2, on='word', how='outer')
    # Fill NaN values with 0
    merged_df = merged_df.fillna(0)
    merged_df = merged_df.rename(columns={'freq_m': month})
    return merged_df


def adjust_count(count,est_df,month):
    """adjust the count based on estimation"""
    coeff = est_df[est_df['month']==month]
    return count * 30 * coeff['sec_per_hour'].item() * coeff['hour'].item() * coeff['word_per_sec'].item() / 1000000


def accum_count(df):
    """get accum count from the second column"""
    # Get the first column (preserved)
    first_column = df.iloc[:, 0]
    # Get cumulative count starting from the second column
    cumulative_counts = df.iloc[:, 1:].cumsum(axis=1)
    # Concatenate the first column with the cumulative counts
    result_df = pd.concat([first_column, cumulative_counts], axis=1)
    return result_df

def load_csv(file_path,start_column):
    # Read the CSV file starting from the given column header
    data = pd.read_csv(file_path)
    # Get the index of the start column
    start_column_index = data.columns.get_loc(start_column)
    # Extract the columns starting from the specified column
    selected_data = data.iloc[:, start_column_index:]
    return selected_data


def apply_threshold(df,threshold:int):
    """apply threhold to all the adjusted counts with word as index"""
    first_column = df.iloc[:, 0]
    # Apply threshold starting from the second column
    thresholded_df = df.iloc[:, 1:].applymap(lambda x: 1 if x > threshold else 0)
    # Concatenate the first column with the thresholded DataFrame
    result_df = pd.concat([first_column, thresholded_df], axis=1)
    return result_df
