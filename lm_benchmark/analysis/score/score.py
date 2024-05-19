#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for freq matching and calculating
"""
import pandas as pd
from pathlib import Path
from .score_util import merge_df,adjust_count, accum_count, load_csv,apply_threshold


class MonthCounter:

    """get the monthly info from the concatenated generation/productions"""
    def __init__(self, gen_file: Path, est_file: Path, test_file: Path, count_all_file: Path,
                 count_test_file: Path, header: str, threshold: int):

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
        self._threshold = threshold
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
            # merge the count with previous one
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


    def get_score(self):
        """ Get matched data """
        if self._selected_rows is None:
            self._selected_rows = self.get_count()
        # apply the threshold apply the freq
        score = apply_threshold(self._selected_rows,self._threshold)
        return score

