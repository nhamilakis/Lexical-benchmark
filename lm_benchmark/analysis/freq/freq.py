#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for freq matching and calculating
"""
import os
import re
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from collections import Counter
from .freq_util import get_freq_table,get_bin_stat,get_equal_bins,match_bin_range,is_word

WORD_PATTERN = re.compile(r'\b\w+\b')

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



class TokenCount:
    def __init__(self, data=None, name=None, header=None):
        if data is not None:
            self.df = pd.DataFrame(list(data.items()), columns=['Word', 'Count']).sort_values(by='Count')
            self.name = name
        else:
            self.df = pd.DataFrame(columns=['Word', 'Count'])
        self.df['Correct']=self.df['Word'].apply(is_word)
        #self.df.set_index('Word', inplace=True)

    def __str__(self):
        return self.df.to_string()

    def __repr__(self):
        return self.df.to_string()

    @staticmethod
    def from_df1(df, name=None):
        assert isinstance(df, pd.DataFrame)
        assert 'Count' in df.columns
        assert 'Word' in df.index.names
        word_count_dict = df['Count'].to_dict()
        return TokenCount(word_count_dict, name)

    def from_df(file_path:str,header:str, name=None):
        lines = pd.read_csv(file_path)[header]
        word_counter = Counter()
        for line in lines:
            words = WORD_PATTERN.findall(line.lower())
            word_counter.update(words)
        return TokenCount(word_counter, header)

    @staticmethod
    def from_text_file(file_path):
        # Read from a text file and count words
        word_counter = Counter()
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # remove all blanks
                line = line.replace(" ", "")
                words = WORD_PATTERN.findall(line.lower())
                word_counter.update(words)
        basename = os.path.splitext(os.path.basename(file_path))[0]
        return TokenCount(word_counter, basename)


    def nonword(self):
        # Find the nonwords
        nonword_df = self.df[self.df['Correct']==False]
        return TokenCount.from_df(nonword_df)


    def difference(self, othercorpus):
        # Find the words in df_ref that are not in df_gen using set difference
        missing_words = self.df.index.difference(othercorpus.df.index)

        # Extract the subset of df_ref with the missing words
        missing_words_df = self.df.loc[missing_words]
        # print("lengh df",len(self.df),"length other",len(othercorpus.df),"length difference",len(missing_words))
        return TokenCount.from_df(missing_words_df)

    def nb_of_types(self):
        # Return the number of unique words (types)
        return self.df.shape[0]

    def nb_of_tokens(self):
        # Return the sum of all word counts (nb of tokens=corpus size)
        return self.df['Count'].sum()

    def zipf_coef(self):
        """Compute the zipf coefficient of a given token count."""
        sorted_data = np.sort(self.df["Count"])
        nbpoints = sorted_data.shape[0]
        x = np.arange(1, (nbpoints + 1))  # ranks
        y = sorted_data[::-1]  # counts
        log_x = np.log(x)
        log_y = np.log(y)
        # Fit a linear regression model in the log-log space
        weights = 1 / x
        wls_model = sm.WLS(log_y, sm.add_constant(log_x), weights=weights)
        results = wls_model.fit()
        intercept = results.params[0]
        slope = results.params[1]
        log_y_fit = results.fittedvalues
        return log_x, log_y_fit, intercept, slope

    def stats(self):
        """Simple descriptive Statistics of the TokenCount (type/token, etc)"""
        print(self.df['Correct'] == False)
        if self.nb_of_tokens() != 0:
            typetok = self.nb_of_types() / self.nb_of_tokens()
        else:
            typetok = np.nan
        d = {'name': self.name, 'nb_token': self.nb_of_tokens(), 'nb_type': self.nb_of_types(), 'type/token': typetok}
        if self.nb_of_types() == 0:
            return d
        nb_hapaxes = np.sum(self.df['Count'] == 1)
        nb_dipaxes = np.sum(self.df['Count'] == 2)
        nb_le10 = np.sum(self.df['Count'] <= 10)
        nb_nonwords = np.sum(self.df['Correct'] == False)
        d1 = {'nb_hapaxes': nb_hapaxes, 'p_hapaxes': nb_hapaxes / self.nb_of_types()}
        d2 = {'nb_dipaxes': nb_dipaxes, 'p_dipaxes': nb_dipaxes / self.nb_of_types()}
        d3 = {'nb_le_10': nb_le10, 'p_le_10': nb_le10 / self.nb_of_types()}
        sorted_data = np.sort(self.df["Count"])
        top_count = sorted_data[-1]
        top_ge10_count = np.sum(sorted_data[-11:-1])
        d4 = {'prop_topcount': top_count / self.nb_of_tokens(),
              'prop_top_ge10_count': top_ge10_count / self.nb_of_tokens()}
        d5 = {'zipf_c': self.zipf_coef()[3]}
        d6 = {'nb_nonwords': nb_nonwords, 'p_nonwords': nb_nonwords / self.nb_of_types()}
        return {**d, **d1, **d2, **d3, **d4, **d5, **d6}






class FreqMatcher1:
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
