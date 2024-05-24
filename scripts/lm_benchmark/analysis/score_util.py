#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path
from lm_benchmark.utils import TokenCount


################################################################################################
# functions for MonthCounter class
#################################################################################################

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



################################################################################################
# MonthCounter class to weight count b yestimation #
#################################################################################################


class MonthCounter:

    """get the monthly info from the concatenated generation/productions"""
    def __init__(self, gen_file: Path, est_file: Path, test_file: Path, count_all_file: Path,
                 count_test_file: Path, header: str):

        if not gen_file.is_file():
            raise ValueError(f'Given file ::{gen_file}:: does not exist !!')
        if not est_file.is_file():
            raise ValueError(f'Given file ::{est_file}:: does not exist !!')
        if not test_file.is_file():
            raise ValueError(f'Given file ::{test_file}:: does not exist !!')
        if not count_all_file.is_file():
            self._merged_df = None     # initialize the merged_all as None if it doesn't exist
            print(f'Count corpus does not exist, creating and saving it to {count_all_file}')
        else:
            print(f'Found count corpus from: {count_all_file}, loading ...')
            self._merged_df = load_csv(count_all_file,'word')

        if not count_test_file.is_file():
            self._selected_rows = None    # initialize the merged_all as None if it doesn't exist
            print(f'Test count does not exist, creating and saving it to {count_test_file}')
        else:
            print(f'Found test count from: {count_test_file}, loading ...')
            self._selected_rows = load_csv(count_test_file,'word')

        self._generation_csv_location = gen_file
        self._estimation_csv_location = est_file
        self._all_csv_location = count_all_file
        self._count_filtered_location = count_test_file
        self._test_csv_location = test_file
        self._header = header

        # Call load method to initialize dataframes
        self.__load__()

    def __load__(self):
        """ Load the dataset into dataframes """
        self._generation_df = pd.read_csv(self._generation_csv_location)
        self._estimation_df = pd.read_csv(self._estimation_csv_location)
        self._test_df = load_csv(self._test_csv_location,'word')

    def __adjusted_count_all__(self):
        """ Match two freq frames """
        # loop over different months
        self._gen_grouped = self._generation_df.groupby('month')
        self._merged_df = pd.DataFrame(columns=['word', 'freq_m'])
        for month, gen_month in self._gen_grouped:
            # get freq in the given month and merge adjusted the count with previous one
            self._merged_df = merge_df(self._merged_df, gen_month, self._header, month)
            # try to rename the initial months' header
            try:
                self._merged_df = self._merged_df.rename(columns={'freq_m_y': month})
            except:
                pass
            # adjust count based on estimation
            self._merged_df[month] = self._merged_df[month].apply(lambda x: adjust_count(x, self._estimation_df, month))

        # remove useless columns
        self._merged_df = self._merged_df.drop(columns=['freq_m_x'])
        # get cumulative frequency
        self._merged_df = accum_count(self._merged_df)
        self._merged_df.to_csv(self._all_csv_location)
        return self._merged_df


    def get_count(self):
        """ Get matched data """
        if self._merged_df is None:
            self._merged_df = self.__adjusted_count_all__()
        # filter the test set
        self._selected_rows = self._merged_df[self._merged_df['word'].isin(self._test_df['word'])]
        self._selected_rows.to_csv(self._count_filtered_location)




