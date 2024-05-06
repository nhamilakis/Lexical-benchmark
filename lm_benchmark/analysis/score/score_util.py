#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from ..freq.freq_util import get_freq_table

def merge_df(merged_df, df2, header: str,month:int):    # TODO: preserve the month group info
    # Merge freq dataframes on 'word' column
    df2 = get_freq_table(df2[header])[['word', 'freq_m']]
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


