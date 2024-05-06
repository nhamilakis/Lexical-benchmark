#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for freq matching and calculating
"""
import pandas as pd
from pathlib import Path
from .score_util import merge_df,adjust_count  # TODO: check how to load this: put in the various fun


class MonthCounter:
    def __init__(self, gen_file: Path, est_file: Path, header: str, threshold: int):

        if not gen_file.is_file():
            raise ValueError(f'Given file ::{gen_file}:: does not exist !!')
        if not est_file.is_file():
            raise ValueError(f'Given file ::{est_file}:: does not exist !!')

        self._generation_csv_location = gen_file
        self._estimation_csv_location = est_file
        self._header = header
        self._threshold = threshold
        self._merged_df = None
        # Call load method to initialize dataframes
        self.__load__()

    def __load__(self) -> None:
        """ Load the dataset into dataframes """
        # filter subset of the words
        self._generation_df = pd.read_csv(self._generation_csv_location)
        self._estimation_df = pd.read_csv(self._estimation_csv_location)


    def __adjusted_count__(self):
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
        self._merged_df.set_index('word', inplace=True)
        # get cumulative frequency
        self._merged_df = self._merged_df.cumsum(axis=1)
        return self._merged_df

    def __estimate_score__(self):
        """ estimate score based on different threholds"""
        # check the existence of the adjusted count

        # apply threshold on the whole dataframe

        return self._matched_CDI

    def count_by_month(self):
        """ Get matched data """
        if self._merged_df is None:
            self._merged_df = self.__adjusted_count__()
        return self._merged_df




