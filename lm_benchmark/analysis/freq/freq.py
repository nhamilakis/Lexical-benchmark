#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for freq matching and calculating
"""

import pandas as pd
from pathlib import Path
from .freq_util import get_freq_table,get_bin_stat,get_equal_bins,match_bin_range


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
    def freq(self) -> pd.DataFrame:
        """ Get the Gold data as a Pandas DataFrame """
        if self._target_df is None:
            self._target_df = self.__build_freq__()
        return self._target_df

    def __load__(self) -> pd.DataFrame:
        """ Load the dataset into a dataframe """
        df = pd.read_csv(self._raw_csv_location)
        # TODO: configure this future CHILDES_trans formats
        if str(self._raw_csv_location).endswith('CHILDES.csv'):
            print('filtering the parents utterances')
            df = df[(df['speaker'] != 'CHI')]     # only use parents' utt
        else:
            print('No action')
        return df

    def __build_freq__(self) -> pd.DataFrame:
        """ Build the freq dataframe from the given src """
        words = self.df[self._header]
        # get freq
        freq_table = get_freq_table(words)
        return freq_table


class FreqMatcher:
    def __init__(self, human: Path, CHILDES: Path, machine:Path,num_bins:int,header: str,freq_header: str):

        if not human.is_file():
            raise ValueError(f'Given file ::{human}:: does not exist !!')
        if not CHILDES.is_file():
            raise ValueError(f'Given file ::{CHILDES}:: does not exist !!')
        if not machine.is_file():
            raise ValueError(f'Given file ::{machine}:: does not exist !!')
        self._human_csv_location = human
        self._machine_csv_location = machine
        self._CHILDES_csv_location = CHILDES
        self._header = header
        self._freq_header = freq_header     #TODO: replace the column headers into this variable
        self._num_bins = num_bins
        self._src_df = None
        self._matched_CDI = None
        self._matched_audiobook = None
        self._human_stat = None
        self._machine_stat = None
        # Call load method to initialize dataframes
        self.__load__()

    def __load__(self) -> None:
        """ Load the dataset into dataframes """
        # filter subset of the words
        self._machine_df = pd.read_csv(self._machine_csv_location)
        self._human = pd.read_csv(self._human_csv_location)
        self._CHILDES = pd.read_csv(self._CHILDES_csv_location)
        # filter CHILDES by parents' inputs
        self._human_df = self._CHILDES[self._CHILDES[self._header].isin(self._human[self._header])]
        # sort df by the given columns
        self._human_df['freq_m'] = self._human_df['freq_m'].astype(float)
        self._machine_df['freq_m'] = self._machine_df['freq_m'].astype(float)
        self._human_df = self._human_df.sort_values(by='freq_m')
        self._machine_df = self._machine_df.sort_values(by='freq_m')
    def __match_freq__(self):
        """ Match two freq frames """
        # get equal bins
        self._CDI_bins, self._matched_CDI = get_equal_bins(self._human_df['freq_m'], self._human_df, self._num_bins)
        self._matched_CDI, self._matched_audiobook = match_bin_range(self._CDI_bins, self._human_df,
                                                         self._machine_df['freq_m'].tolist(),
                                                         self._machine_df, False)
        self._human_stat = get_bin_stat(self._matched_CDI, 'freq_m')
        self._machine_stat = get_bin_stat(self._matched_audiobook, 'freq_m')
        return self._matched_CDI, self._matched_audiobook, self._human_stat,self._machine_stat

    def get_matched_data(self) -> tuple:
        """ Get matched data """
        if self._matched_CDI is None:
            self._matched_CDI, self._matched_audiobook, self._human_stat,self._machine_stat = self.__match_freq__()
        return self._matched_CDI, self._matched_audiobook, self._human_stat,self._machine_stat
