#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
util func for get_stat

@author: jliu
"""
import string
import pandas as pd
import math
import numpy as np
import collections
import matplotlib.pyplot as plt
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from itertools import combinations
import re
import os 


def phonememize(word):
 
     '''
     phonemize single word/utterance
     input: grapheme string; output: phonemized string
     '''
     
     backend = EspeakBackend('en-us', language_switch="remove-utterance")
     separator = Separator(word=None, phone=" ")
     phonemized = backend.phonemize([word], separator=separator)[0].strip()
     return phonemized
 
    
def get_freq_table(lines):
    
    '''
    clean each .cha transcript
    input: 1. path to the .cha transcripts from CHILDES
           2. receotive or expressive vocab(related with the speaker)
    ouput: [the cleaned transcript],[the word list of the cleaned transcript]
    '''
    
    def get_freq(result):
        
        '''
        input: raw .txt file
        output: the freq dataframe with all the words and their raw freq
        '''
        # clean text and get the freq table
        frequencyDict = collections.Counter(result)  
        freq_lst = list(frequencyDict.values())
        word_lst = list(frequencyDict.keys())
        
        # get freq
        fre_table = pd.DataFrame([word_lst,freq_lst]).T
        col_Names=["Word", "Freq"]
        fre_table.columns = col_Names
        fre_table['Norm_freq'] = fre_table['Freq']/len(result)
        fre_table['Norm_freq_per_million'] = fre_table['Norm_freq']*1000000
        # get log_freq
        log_freq_lst = []
        for freq in freq_lst:
            log_freq = math.log10(freq)
            log_freq_lst.append(log_freq)
        fre_table['Log_freq'] = log_freq_lst
        
        # get logarithm of normalized word freq per million
        norm_log_freq_lst = []
        for freq in fre_table['Norm_freq_per_million'].tolist():
            norm_log_freq = math.log10(freq)
            norm_log_freq_lst.append(norm_log_freq)
        fre_table['Log_norm_freq_per_million'] = norm_log_freq_lst
        
        return fre_table

    
    
        # Remove empty lines using a list comprehension
    non_empty_lines_lst = [line for line in lines if line.strip()]
    word_lst = []
    for script in non_empty_lines_lst: 
            # remove annotations
            translator = str.maketrans('', '', string.punctuation+ string.digits)
            clean_string = script.translate(translator).lower()
            cleaned = re.sub(' +', ' ', clean_string.strip())
            # get the word lst
            words = cleaned.split(' ')
            word_lst.extend(words)
            
    fre_table = get_freq(word_lst)
    
    return fre_table



def convert_to_log(freq):
    return math.log10(freq)





def plot_density_hist(data,n_bins,label,alpha=0.5):
      
      # Creating the bins with equal number of observations
      data_sorted = np.sort(data)
      bins = np.array([data_sorted[i] for i in range(0, len(data_sorted), len(data_sorted)//n_bins)])
    
      # Making sure to include the maximum value in the last bin
      if bins[-1] != data_sorted[-1]:
          bins = np.append(bins, data_sorted[-1])
    
      # Plotting the histogram
      plt.hist(data, bins=bins, density=True, alpha=alpha,edgecolor='black',label=label)
      return bins
      

def match_density_hist(data,target_bins,label,alpha=0.5):
    
      
      def find_closest_numbers(arr, target_array):
        closest_numbers = [min(arr, key=lambda x: abs(x - target)) for target in target_array]
        return np.array(closest_numbers)
    
      # Creating the bins with equal number of observations
      data_sorted = np.sort(data)
      # find the corresponding number in the array
      bins = find_closest_numbers(target_bins, data_sorted)
    
      # Making sure to include the maximum value in the last bin
      if bins[-1] != data_sorted[-1]:
          bins = np.append(bins, data_sorted[-1])
    
      # Plotting the histogram
      plt.hist(data, bins=bins, density=True, alpha=alpha,edgecolor='black',label=label)




      

# distribution of matched sets
def plot_single(freq,n,freq_type,lang,eval_condition, eval_type, word_format):  
    
    
    
    if eval_type == 'CDI': 
        
        data = freq['CHILDES_'+ freq_type + '_per_million'].tolist()
    
    else:
        data = freq['Audiobook_'+ freq_type + '_per_million'].tolist()
   
    # plot freq per million     
    plot_density_hist(data,n,label = eval_type)
     
    
    plt.xlabel(freq_type + '_per million')
    plt.ylabel('Density')
    
    # set the limits of the x-axis for each line
    if freq_type == 'freq':
        
        plt.xlim(0,850)
        plt.ylim(0,0.035)
            
    elif freq_type == 'log_freq':
        
        plt.xlim(-1,4)
        plt.ylim(0,2)
    
    plt.legend() 

    

      
def get_overlapping(word_lists_dict):
    
    
    # Get all combinations of list names
    pairs = list(combinations(word_lists_dict.keys(), 2))
    
    # Initialize a dictionary to store counts of common words
    overlap_dict = {}
    
    # Calculate common words count for each pair of lists
    for list1, list2 in pairs:
        common_words = len(set(word_lists_dict[list1]).intersection(word_lists_dict[list2]))
        overlap_dict[(list1, list2)] = common_words
    
    # Get the unique list names
    list_names = list(word_lists_dict.keys())
    
    # Initialize the DataFrame with dashes and set column and row names
    overlap_df = pd.DataFrame('-', index=list_names, columns=list_names)
    
    # Fill the upper half of the DataFrame with counts of common words
    for i in range(len(list_names)):
        for j in range(i + 1, len(list_names)):  # Iterate from i+1 to avoid duplicated pairs
            if i != j:
                value = overlap_dict.get((list_names[i], list_names[j]), '-')
                overlap_df.iloc[i, j] = value
                overlap_df.iloc[j, i] = value  # Fill symmetrically
    
    return overlap_df


def phonemize_frame(fre_table,col_head):
    

    # step 1: phonemize all the words
    fre_table['phon'] = fre_table[col_head].apply(phonememize)
    # step 2: merge similar words 
    
    return fre_table


def get_overlap_audiobook():
    # get the overlapping test sets
    # filter the results with more than 5 variations
    gold_phoneme = pd.read_csv('/data/Lexical-benchmark/recep/wuggy/gold_test.csv')
    freq = pd.read_csv('/data/Lexical-benchmark/recep/frequencies.csv')
    # overlapping
    selected_words = list(set(freq['word'].tolist()).intersection(set(gold_phoneme['word'].tolist())))
    # get overall counts
    selected_audiobook = freq[freq['word'].isin(selected_words)]
    count= selected_audiobook.iloc[:, 2:].sum(axis=1)
    selected_audiobook = selected_audiobook[['word']]
    selected_audiobook['count'] = count
    selected_audiobook.to_csv('Audiobook_test.csv')


def match_stat(freq_path):
    
    # get length and overlapping
    len_dict = {}
    word_lists_dict = {}
    for file in os.listdir(freq_path):
        test = pd.read_csv(freq_path + '/' + file)
        len_dict[file] = test.shape[0]   
        word_lists_dict[file] = test['word'].tolist()
    matched_stat = get_overlapping(word_lists_dict)
    return len_dict,matched_stat
 

