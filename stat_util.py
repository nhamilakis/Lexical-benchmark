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



def match_range(CDI,audiobook):
    
    '''
    match the audiobook sets with CHILDES of differetn modes
    Returns shrinked dataset with the matched range
    '''
    max_freq = min(CDI['CHILDES_freq_per_million'].max(),audiobook['Audiobook_freq_per_million'].max())
    min_freq = max(CDI['CHILDES_freq_per_million'].min(),audiobook['Audiobook_freq_per_million'].min())
    matched_CDI = CDI[(CDI['CHILDES_freq_per_million'] >= min_freq) & (CDI['CHILDES_freq_per_million'] <= max_freq)]
    matched_audiobook = audiobook[(audiobook['Audiobook_freq_per_million'] >= min_freq) & (audiobook['Audiobook_freq_per_million'] <= max_freq)]
    # sort the results by freq 
    matched_CDI = matched_CDI.sort_values(by='CHILDES_freq_per_million')
    matched_audiobook = matched_audiobook.sort_values(by='Audiobook_freq_per_million')
    
    return matched_CDI,matched_audiobook


def get_bin_stat(bins,data_sorted):
    
    '''
    get stat of freq bins
    '''
    # computing statistics over the bins (size, min, max, mean, med, low and high boundaries and density)
    boundaries=list(zip(bins[:-1],bins[1:]))
    binned_data_count=[len(data_sorted[(data_sorted>=l) & (data_sorted<h)]) for l,h in boundaries]
    binned_data_min =[np.min(data_sorted[(data_sorted>=l) & (data_sorted<h)]) for l,h in boundaries]
    binned_data_max=[np.max(data_sorted[(data_sorted>=l) & (data_sorted<h)]) for l,h in boundaries]
    binned_data_mean=[np.mean(data_sorted[(data_sorted>=l) & (data_sorted<h)]) for l,h in boundaries]
    binned_data_median=[np.median(data_sorted[(data_sorted>=l) & (data_sorted<h)]) for l,h in boundaries]
    bins_stats=pd.DataFrame({'count':binned_data_count,'min':binned_data_min,'max':binned_data_max,'mean':binned_data_mean,'median':binned_data_median})
    bins_stats['low']=bins[:-1]  
    bins_stats['high']=bins[1:]
    bins_stats['density']=bins_stats['count']/(bins_stats['high']-bins_stats['low'])/sum(bins_stats['count'])
    
    return bins_stats
    


def get_equal_bins(data,n_bins):
    
    '''
    get equal-sized bins
    input: a sorted array or a list of numbers; computes a split of the data into n_bins bins of approximately the same size
    
    return
        bins: array with each bin boundary
        bins_stats  
    '''
    # preparing data (adding small jitter to remove ties)
    size=len(data)
    assert n_bins<=size,"too many bins compared to data size"
    mindif=np.min(np.abs(np.diff(np.sort(np.unique(data))))) # minimum difference between consecutive distinct values
    jitter=mindif*0.01  # this small jitter will not change the relative order between datapoints
    data_jitter=np.array(data)+np.random.uniform(low=-jitter, high=jitter, size=size)
    data_sorted = np.sort(data_jitter) # little jitter to remove ties

    # Creating the bins with approx equal number of observations
    bin_indices = np.linspace(1, len(data), n_bins+1)-1   # indices to edges in sorted data
    bins=[data_sorted[0]] # left edge inclusive
    bins=np.append(bins,[(data_sorted[int(b)]+data_sorted[int(b+1)])/2 for b in bin_indices[1:-1]])
    bins = np.append(bins, data_sorted[-1]+jitter)  # this is because the extreme right edge is inclusive in plt.hits
    
    
    # computing bin membership for the original data; append bin membership to stat
    bin_membership=np.zeros(len(bins),dtype=int)
    for i in range(0,len(bins)-1):
       bin_membership[(data_sorted>=bins[i])&(data_sorted<bins[i+1])]=i
    
    data_sorted['group_' + str(len(bins))] = bin_membership
    
    return bins, data_sorted 



def match_bin_range(CDI_bins,audiobook):
    
    '''
    match range of the audiobook freq of machine CDI with CHILDES freq of CDI
    
    input: 
        human CDI eauql-sized bins
        machine-CDI freq bin
    Returns 
        bins: machiine CDI with adjusted group array
        bins_stats: machine CDI dataframe with annotated group
    '''
    
    def find_closest_numbers(arr, target_array):
        closest_numbers = [min(arr, key=lambda x: abs(x - target)) for target in target_array]
        return np.array(closest_numbers)
     
    # Creating the bins with approx equal number of observations
    bins = find_closest_numbers(audiobook, CDI_bins)
    
    # computing bin membership for the original data; append bin membership to stat
    bin_membership=np.zeros(len(bins),dtype=int)
    for i in range(0,len(bins)-1):
       bin_membership[(audiobook>=bins[i])&(audiobook<bins[i+1])]=i
    
    audiobook['group_' + str(len(bins))] = bin_membership
       
    return bins, audiobook 




def match_bin_density(CDI_bins_stats,audiobook_bins_stats,audiobook_data):
    
    '''
    match density of the audiobook freq of machine CDI with CHILDES freq of CDI
    
    input: 
        human CDI eauql-sized bins
        machine-CDI freq bin
    Returns 
        bins: machiine CDI with adjusted group array
        bins_stats: machine CDI dataframe with annotated group
    '''
    # adjust based on density
    CDI_density = set(CDI_bins_stats['density'].tolist())
    audiobook_density = set(audiobook_bins_stats['density'].tolist())
    
    # divide by density
    result_set = [a for a, b in zip(CDI_density, audiobook_density) if a > b]
    
    
    # get the selected rows
    remained_rows = 
    rows_to_reduce = 
    
    # sort the difference
    
    # the objective would be the smallest overall density dist
    density_diff = 0
    
    
    return bins, bins_stats, matched_audiobook


def plot_density_hist(data,n_bins,label,alpha=0.5):
  """Takes as input an array or a list of numbers; computes a split of the data into n_bins bins of approximately the same size
     taking into account ties. Returns descriptive stats for each bins, a membership array (tells which bins each datapoint belongs to)
     and plots a quantile density histogram"""
  # preparing data (adding small jitter to remove ties)
  size=len(data)
  assert n_bins<=size,"too many bins compared to data size"
  mindif=np.min(np.abs(np.diff(np.sort(np.unique(data))))) # minimum difference between consecutive distinct values
  jitter=mindif*0.01  # this small jitter will not change the relative order between datapoints
  data_jitter=np.array(data)+np.random.uniform(low=-jitter, high=jitter, size=size)
  data_sorted = np.sort(data_jitter) # little jitter to remove ties

  # Creating the bins with approx equal number of observations
  bin_indices = np.linspace(1, len(data), n_bins+1)-1   # indices to edges in sorted data
  bins=[data_sorted[0]] # left edge inclusive
  bins=np.append(bins,[(data_sorted[int(b)]+data_sorted[int(b+1)])/2 for b in bin_indices[1:-1]])
  bins = np.append(bins, data_sorted[-1]+jitter)  # this is because the extreme right edge is inclusive in plt.hits
    
  # computing statistics over the bins (size, min, max, mean, med, low and high boundaries and density)
  boundaries=list(zip(bins[:-1],bins[1:]))
  binned_data_count=[len(data_sorted[(data_sorted>=l) & (data_sorted<h)]) for l,h in boundaries]
  binned_data_min =[np.min(data_sorted[(data_sorted>=l) & (data_sorted<h)]) for l,h in boundaries]
  binned_data_max=[np.max(data_sorted[(data_sorted>=l) & (data_sorted<h)]) for l,h in boundaries]
  binned_data_mean=[np.mean(data_sorted[(data_sorted>=l) & (data_sorted<h)]) for l,h in boundaries]
  binned_data_median=[np.median(data_sorted[(data_sorted>=l) & (data_sorted<h)]) for l,h in boundaries]
  bins_stats=pd.DataFrame({'count':binned_data_count,'min':binned_data_min,'max':binned_data_max,'mean':binned_data_mean,'median':binned_data_median})
  bins_stats['low']=bins[:-1]  
  bins_stats['high']=bins[1:]
  bins_stats['density']=bins_stats['count']/(bins_stats['high']-bins_stats['low'])/sum(bins_stats['count'])

   
  # computing bin membership for the original data
  bin_membership=np.zeros(size,dtype=int)
  for i in range(0,len(bins)-1):
     bin_membership[(data_jitter>=bins[i])&(data_jitter<bins[i+1])]=i

  # Plotting the histogram
  plt.hist(data_sorted, bins=bins, density=True, alpha=alpha,edgecolor='black',label=label)

  return bins, bins_stats  



    

      
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
 

def filter_words(gold_path):
    
    '''
    this won't matter: as we have similar words anyway
    
    '''
    # check whether the selected words carry 5 nonword pairs 
    
    gold = pd.read_csv(gold_path)
    # check the number of phoneme variations   the id would be theversion of the 
    
    unique_df = gold.drop_duplicates(subset='id')
    
    # Group by column 'B' and count the number of rows in each group
    grouped_df = unique_df.groupby(['word']).size().reset_index(name='Count')
    
    # Convert the grouped DataFrame to a dictionary
    result_dict = dict(zip(grouped_df['word'], grouped_df['Count']))
    
    filtered_dict = {key: value for key, value in result_dict.items() if value >= 6}
    filtered_frame = pd.DataFrame(list(filtered_dict.items()), columns=['word', 'pair'])
    # output the filtered 
    filtered_frame.to_csv('/data/Lexical-benchmark/stat/corpus/char/Audiobook_test.csv')
    return filtered_frame


def phonememize(word):
 
     '''
     phonemize single word/utterance
     input: grapheme string; output: phonemized string
     '''
     
     backend = EspeakBackend('en-us', language_switch="remove-utterance")
     separator = Separator(word=None, phone=" ")
     phonemized = backend.phonemize([word], separator=separator)[0].strip()
     return phonemized
 
    