#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for freq matching and calculating
"""

import pandas as pd
from pathlib import Path
from ..utils import get_freq_table


class FreqGenerater:

    def __init__(self, raw_csv: Path, header: str):

        if not raw_csv.is_file():
            raise ValueError(f'Given file ::{raw_csv}:: does not exist !!')
        self._raw_csv_location = raw_csv
        self._header = header

        # Zero init
        self._src_df = None
        self._target_df = None

    @property
    def df(self) -> pd.DataFrame:
        """ Get the data as a Pandas DataFrame """
        if self._src_df is None:
            self._src_df = self.__load__()
        return self._src_df

    @property
    def gold(self) -> pd.DataFrame:
        """ Get the Gold data as a Pandas DataFrame """
        if self._target_df is None:
            self._target_df = self.__build_freq__()
        return self._target_df

    def __load__(self) -> pd.DataFrame:
        """ Load the dataset into a dataframe """
        df = pd.read_csv(self._raw_csv_location)
        return df

    def __build_freq__(self) -> pd.DataFrame:
        """ Build the freq dataframe from the given src """
        word_lst = self.df[self._header]
        words = [str(word) for sentence in word_lst for word in str(sentence).split()]
        # get freq
        freq_table = get_freq_table(words)
        return freq_table

