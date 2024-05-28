#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common util func for all the packages
"""
import os
import re
import pandas as pd
import numpy as np
import statsmodels.api as sm
from collections import Counter

import enchant

WORD_PATTERN = re.compile(r'\b\w+\b')
d_uk = enchant.Dict("en_UK")
d_us = enchant.Dict("en_US")



def is_word(word):
    # Function to check if a word is valid
    true_word = ["cant", "wont", "dont", "isnt", "its", "im", "hes", "shes", "theyre", "were", "youre", "lets",
                 "wasnt", "werent", "havent", "ill", "youll", "hell", "shell", "well", "theyll", "ive", "youve",
                 "weve", "theyve", "shouldnt", "couldnt", "wouldnt", "mightnt", "mustnt", "thats", "whos", "whats", "wheres", "whens", "whys", "hows", "theres", "heres", "lets", "wholl", "whatll", "whod", "whatd", "whered", "howd", "thatll", "whatre", "therell", "herell"]
    try:
        if d_uk.check(word) or d_us.check(word) or d_us.check(word.capitalize()) or d_uk.check(word.capitalize()) or word in true_word:
            return True
        else:
            return False
    except:
        return False



class TokenCount:
    def __init__(self, data=None, name=None, header=None):
        if data is not None:
            self.df = pd.DataFrame(list(data.items()), columns=['word', 'count']).sort_values(by='count')
            self.name = name
        else:
            self.df = pd.DataFrame(columns=['word', 'count'])
        # check the existence of the columns below
        self.df['freq_m'] = self.df['count']/self.df['count'].sum() * 1000000
        self.df['correct']=self.df['word'].apply(is_word)
        #self.df.set_index('word', inplace=True)

    def __str__(self):
        return self.df.to_string()

    def __repr__(self):
        return self.df.to_string()

    @staticmethod
    def from_df(file_path,header:str, name=None):
        try:
            lines = pd.read_csv(file_path)[header]
        except:   # in the case that the input is already a dataframe
            lines = file_path[header]
        # remove nan in the column
        lines = lines.dropna()
        lines = lines.astype(str)
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
        nonword_df = self.df[self.df['correct']==False]
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
        return self.df['count'].sum()

    def zipf_coef(self):
        """Compute the zipf coefficient of a given token count."""
        sorted_data = np.sort(self.df["count"])
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
        print(self.df['correct'] == False)
        if self.nb_of_tokens() != 0:
            typetok = self.nb_of_types() / self.nb_of_tokens()
        else:
            typetok = np.nan
        d = {'name': self.name, 'nb_token': self.nb_of_tokens(), 'nb_type': self.nb_of_types(), 'type/token': typetok}
        if self.nb_of_types() == 0:
            return d
        nb_hapaxes = np.sum(self.df['count'] == 1)
        nb_dipaxes = np.sum(self.df['count'] == 2)
        nb_le10 = np.sum(self.df['count'] <= 10)
        nb_nonwords = np.sum(self.df['correct'] == False)
        d1 = {'nb_hapaxes': nb_hapaxes, 'p_hapaxes': nb_hapaxes / self.nb_of_types()}
        d2 = {'nb_dipaxes': nb_dipaxes, 'p_dipaxes': nb_dipaxes / self.nb_of_types()}
        d3 = {'nb_le_10': nb_le10, 'p_le_10': nb_le10 / self.nb_of_types()}
        sorted_data = np.sort(self.df["count"])
        top_count = sorted_data[-1]
        top_ge10_count = np.sum(sorted_data[-11:-1])
        d4 = {'prop_topcount': top_count / self.nb_of_tokens(),
              'prop_top_ge10_count': top_ge10_count / self.nb_of_tokens()}
        d5 = {'zipf_c': self.zipf_coef()[3]}
        d6 = {'nb_nonwords': nb_nonwords, 'p_nonwords': nb_nonwords / self.nb_of_types()}
        return {**d, **d1, **d2, **d3, **d4, **d5, **d6}




