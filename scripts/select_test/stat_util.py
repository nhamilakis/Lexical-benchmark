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
#from phonemizer.backend import EspeakBackend
#from phonemizer.separator import Separator
import os 
import matplotlib.pyplot as plt
import spacy
import re
import random
from itertools import islice
from collections import deque

random.seed(66)
nlp = spacy.load('en_core_web_sm')


def preprocess(infants_data,CDI_type):
    '''
    clean the human CDI testset

        input: word list
        output: selected word freq dataframe
    '''

    if CDI_type == 'human':

        # remove annotations or additional punctuations
        words = infants_data['item_definition'].tolist()

        cleaned_lst = []
        for word in words:
            # remove punctuations
            translator = str.maketrans('', '', string.punctuation + string.digits)
            clean_string = word.translate(translator).lower()
            # remove annotations; problem: polysemies
            cleaned_word = re.sub(r"\([a-z]+\)", "", clean_string)
            cleaned_lst.append(cleaned_word)

        infants_data['word'] = cleaned_lst
    return infants_data


def select_type(selected_words, word_type):

    '''
    select words based on POS
    input: dataframe with column [words]
    return dataframe with selected words adn POS tags
    '''
    # select open class words
    pos_all = []
    for word in selected_words['word'].tolist():
        doc = nlp(word)
        pos_lst = []
        for token in doc:
            pos_lst.append(token.pos_)
        pos_all.append(pos_lst[0])
    selected_words['POS'] = pos_all

    content_POS = ['ADJ', 'NOUN', 'VERB', 'ADV', 'PROPN']
    if word_type == 'all':
        selected_words = selected_words
    elif word_type == 'content':
        selected_words = selected_words[selected_words['POS'].isin(content_POS)]
    elif word_type == 'function':
        selected_words = selected_words[~selected_words['POS'].isin(content_POS)]

    return selected_words


def get_ttr(freq_table):
    
    '''
    get the type-to-token ratios
    '''
    return freq_table.shape[0]/freq_table['Freq'].sum() * 100



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
        translator = str.maketrans('', '', string.punctuation + string.digits)
        clean_string = script.translate(translator).lower()
        cleaned = re.sub(' +', ' ', clean_string.strip())
        # get the word lst
        words = cleaned.split(' ')
        word_lst.extend(words)

    fre_table = get_freq(word_lst)

    return fre_table



def get_intersections(df1,df2,column1,column2):
    
    #align dataframe 1 with dataframe 2
    max_freq = min(df1[column1].max(), df2[column2].max())
    min_freq = max(df1[column1].min(), df2[column2].min())
    matched_df1 = df1[(df1[column1] >= min_freq) & (df1[column1] <= max_freq)]
    matched_df2 = df2[(df2[column2] >= min_freq) & (df2[column2] <= max_freq)]

    return matched_df1,matched_df2



  
def match_medians(word_freq_dict, target_median, target_len):
    
    '''
    match the freq and len median while preserving their range
    
    add another constraints on len: median
    
    operate on dict bc we need to take word as index
    
    return a selected nested list candidate
    '''
    
    def find_closest_number(lst, target):
        closest_number = lst[0]
        min_difference = abs(target - lst[0])
        
        for number in lst:
            difference = abs(target - number)
            if difference < min_difference:
                min_difference = difference
                closest_number = number
                
        return closest_number
    
    
    
    def divide_dict(candi_dict,target_len,condi):
        
        '''
        divide a dictionary based on target medians
        input: unsorted dictionary
        return two dictionaries
        '''
        if condi == 'freq':
            sorted_candi_dict = dict(sorted(candi_dict.items(), key=lambda item: item[1][0]))
            # get the closest num to target word
            rest_len_sorted = [length for length, _ in sorted_candi_dict.values()]
            
        else:
            sorted_candi_dict = dict(sorted(candi_dict.items(), key=lambda item: item[1][1]))
            # get the closest num to target word
            rest_len_sorted = [length for _, length in sorted_candi_dict.values()]
        
        # get the closest len num; assume it's in the rest of the list
        closest_number = find_closest_number(rest_len_sorted, target_len)
        # initialize left and right dict in the fixed dict
        index = rest_len_sorted.index(closest_number)
        
        left_dict = dict(islice(sorted_candi_dict.items(), index))
        right_dict = dict(deque(sorted_candi_dict.items(), maxlen=len(rest_len_sorted)-index))
        
        return left_dict,right_dict,index
    
    
    
    def generate_index(candi_dict,rest_dict,target_len,num):
        
        '''
        input: rest of index, index lst to select from, num, range_index
        return a list of selected index 
        '''
        
        def select_index(candi_dict,index):
            
            def shuffle_dict(my_dict):
                # Get a list of key-value tuples and shuffle it
                items = list(my_dict.items())
                random.shuffle(items)
                # Convert the shuffled list back to a dictionary
                shuffled_dict = dict(items)
                return shuffled_dict
            
            candi_dict = shuffle_dict(candi_dict)
            
            selected_dict = dict(islice(candi_dict.items(), index))
            return selected_dict
        
        # divide the fixed rest dictionary into 2 parts
        left_dict_rest,right_dict_rest,_ = divide_dict(rest_dict,target_len,'len')
        left_dict_candi,right_dict_candi,_ = divide_dict(candi_dict,target_len,'len')
        # select the results based on fixed num
        added_num = abs(len(left_dict_rest) - len(right_dict_rest))
        # equally allocate the rest of index
        mod = (num - added_num) % 2
        allo_num = (num - added_num) // 2
            
        if len(left_dict_rest) >= len(right_dict_rest):      
            # select the index to align right side
            right_candi_num = added_num + allo_num + mod
            left_candi_num = added_num
            
        elif len(right_dict_rest) > len(left_dict_rest):
            # select the index from left side
            left_candi_num = added_num + allo_num + mod
            right_candi_num = added_num
            
        # select from candi dict based on the number of index 
        selected_left = select_index(left_dict_candi,left_candi_num)
        selected_right = select_index(right_dict_candi,right_candi_num)
        rest_dict.update(selected_left) 
        rest_dict.update(selected_right) 
        return rest_dict
    

    def get_range_dict(word_freq_dict):
        
        items = list(word_freq_dict.items())
        range_dict = {}
        len_lst = [length for _, length in word_freq_dict.values()]
        freq_lst = [freq for freq, _ in word_freq_dict.values()]
        
        # get range of freq and len lists
        min_len_index = len_lst.index(min(len_lst))
        max_len_index = len_lst.index(max(len_lst))
        min_freq_index = freq_lst.index(min(freq_lst))
        max_freq_index = freq_lst.index(max(freq_lst))
        
        # get key-value pairs
        range_dict[items[min_len_index][0]]=items[min_len_index][1]
        range_dict[items[max_len_index][0]]=items[max_len_index][1]
        range_dict[items[min_freq_index][0]]=items[min_freq_index][1]
        range_dict[items[max_freq_index][0]]=items[max_freq_index][1]
        
        return range_dict

    # initialize the dict sorted by freq 
    word_freq_dict = dict(sorted(word_freq_dict.items(), key=lambda item: item[1][0]))
    left_dict,right_dict,index = divide_dict(word_freq_dict,target_median,'freq')
    range_dict = get_range_dict(word_freq_dict)
    
    # update left or right dict
    if index > len(word_freq_dict)-index:
        print('remove indices in the left')
        # remove range datapoints from candi dict
        for key in range_dict:
            left_dict.pop(key, None)
        right_dict.update(range_dict)
        updated_dict = generate_index(left_dict,right_dict,target_len,len(word_freq_dict)-index)   
        
    else:
        print('remove indices in the right')
        # put range index into the fixed dict
        for key in range_dict:
            right_dict.pop(key, None)
        left_dict.update(range_dict)
        updated_dict = generate_index(right_dict,left_dict,target_len,index)
     
    return updated_dict




def match_range(CDI,audiobook):
    '''
    match the audiobook sets with CHILDES of differetn modes
    Returns shrinked dataset with the matched range
    '''
    matched_CDI,matched_audiobook = get_intersections(CDI,audiobook,'CHILDES_log_freq_per_million','Audiobook_log_freq_per_million')

    # sort the results by freq 
    matched_CDI = matched_CDI.sort_values(by='CHILDES_log_freq_per_million')
    matched_audiobook = matched_audiobook.sort_values(by='Audiobook_log_freq_per_million')
    
    return matched_CDI,matched_audiobook


def get_bin_stat(bins,data_sorted):
    
    '''
    get stat of freq bins
    input: column with annotated group name
    return bin_stat
    '''
    data_sorted = np.array(data_sorted)
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
    # Rename the newly created column to 'group'
    bins_stats = bins_stats.reset_index()
    bins_stats = bins_stats.rename(columns={'index': 'group'})

    return bins_stats
    

def get_len_stat(df,column_header):
    '''
    get stat of word length
    input: dataframe with annotated group name
    return bin_stat
    '''
    # Group by 'Group' column and calculate statistics
    stats_df = df.groupby('group')[column_header].agg(
        min='min',
        max='max',
        mean='mean',
        median='median'
    ).reset_index()

    stats_df.rename(columns={'min': 'len_min','max': 'len_max','mean': 'len_mean','median': 'len_median'}, inplace=True)
    return stats_df



def get_equal_bins(data,data_frame,n_bins):
    
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
    bin_membership=np.zeros(size,dtype=int)
    for i in range(0,len(bins)-1):
        bin_membership[(data_jitter>=bins[i])&(data_jitter<bins[i+1])]=i
    
    data_frame['group'] = bin_membership
    
    return bins, data_frame


def match_bin_range(CDI_bins,CDI,audiobook,audiobook_frame,match_median):
    
    '''
    match range of the audiobook freq of machine CDI with CHILDES freq of CDI
    
    input: 
        human CDI eauql-sized bins
        machine-CDI freq bin
    Returns 
        bins: machiine CDI with adjusted group array
        bins_stats: machine CDI dataframe with annotated group
    '''

    def align_group(CDI,audiobook_frame):

        matched_CDI = pd.DataFrame()
        matched_audiobook = pd.DataFrame()

        for group in set(audiobook_frame['group']):
            CDI_group = CDI[CDI['group'] == group]
            audiobook_group = audiobook_frame[audiobook_frame['group'] == group]
            CDI_selected, audiobook_selected = get_intersections(CDI_group, audiobook_group, 'word_len', 'word_len')
            matched_CDI = pd.concat([matched_CDI, CDI_selected])
            matched_audiobook = pd.concat([matched_audiobook, audiobook_selected])

        return matched_CDI, matched_audiobook

    def find_closest_numbers(arr, target_array):
        closest_numbers = [min(arr, key=lambda x: abs(x - target)) for target in target_array]
        return np.array(closest_numbers)

    audiobook = np.array(audiobook)
    # Creating the bins with approx equal number of observations
    bins = find_closest_numbers(audiobook, CDI_bins)
    # computing bin membership for the original data; append bin membership to stat
    bin_membership=np.zeros(len(audiobook),dtype=int)
    for i in range(0,len(bins)-1):
       bin_membership[(audiobook>=bins[i])&(audiobook<=bins[i+1])]=i

    audiobook_frame['group'] = bin_membership
    
    CDI,audiobook_frame = align_group(CDI,audiobook_frame)
    
    if not match_median:
        return CDI, audiobook_frame

    else:
        # match freq and len medians of each freq bin
        target_frame_grouped = CDI.groupby('group')
        matched_audiobook = pd.DataFrame()   
        
        for group, target_frame_group in target_frame_grouped:
            
            target_freq = target_frame_group['CHILDES_log_freq_per_million'].median()
            
            target_len = target_frame_group['word_len'].median()
            
            machine_group_frame = audiobook_frame[audiobook_frame['group'] == group]
            machine_group = {}
            for _, row in machine_group_frame.iterrows():
                key = row['word']
                values = (row['Audiobook_log_freq_per_million'], row['word_len'])
                machine_group[key] = values
                
            updated_dict = match_medians(machine_group, target_freq,target_len)
            
            frequencyDict = collections.Counter(updated_dict.values()) 
            count_lst = list(frequencyDict.values())
            freq_lst = list(frequencyDict.keys())   
            # Randomize the row orders
            randomized_df = machine_group_frame.sample(frac=1)
            word_frame = pd.DataFrame()   
            n = 0
            while n < len(freq_lst):
                selected_frame_words = randomized_df[randomized_df['Audiobook_log_freq_per_million']==freq_lst[n][0]]
                selected_frame_words = selected_frame_words[selected_frame_words['word_len']==freq_lst[n][1]]
                # generate the index randomly 
                selected_frame_words = selected_frame_words.reindex()
                selected_frame = selected_frame_words.iloc[:count_lst[n]]
                word_frame = pd.concat([word_frame,selected_frame])
                n += 1
           
            matched_audiobook = pd.concat([matched_audiobook,word_frame])
    
        return CDI, matched_audiobook
    
                 
def match_bin_density(matched_CDI,matched_audiobook,CDI_bins,audiobook_bins, threshold):
    
    '''
    match density of the audiobook freq of machine CDI with CHILDES freq of CDI
    
    input: 
        human CDI eauql-sized bins
        machine-CDI freq bin
    Returns 
        bins: machiine CDI with adjusted dataframe
        bins_stats: machine CDI dataframe with annotated group
    '''
    
    # update the results
    def update_bins(CDI_bins_stats,audiobook_bins,matched_audiobook):
        
        '''
        for one update: only remove one median row of the selected band
        return the dictionary and updated audiobook frame
    
        to preserve the maximum num, do not reduce words in the lower density bins
        preserve the largest by pair-wise match
        '''
        
        def get_bin_diff(CDI_bins_stats,audiobook_bins,matched_audiobook):
            audiobook_bins_stats = get_bin_stat(audiobook_bins,matched_audiobook['Audiobook_log_freq_per_million'].tolist())
            # map the result with the diectionary key
            # Merge DataFrames on column group
            merged_df = pd.merge(audiobook_bins_stats, CDI_bins_stats, on='group', suffixes=('_audio', '_CDI'))
            # Create a dictionary with key as the values of column A and value as the difference
            diff_dict = {row['group']: row['density_audio'] - row['density_CDI'] for _, row in merged_df.iterrows()}
            return diff_dict
        
        diff_dict = get_bin_diff(CDI_bins_stats,audiobook_bins,matched_audiobook)
        # Find the key-value pair with the highest value
        max_pair = max(diff_dict.items(), key=lambda x: x[1])
        # in the first iteration, get the lower density bins: do not change this
        unchanged_group = [] 
        for key, value in diff_dict.items():
            if value < 0:
                unchanged_group.append(key)
                
        # resort the diff_dict
        if max_pair[1] > 0 and max_pair[0] not in unchanged_group:
            # remove the median ele
            selected_frame = matched_audiobook[matched_audiobook['group'] == max_pair[0]]
            freq = selected_frame.iloc[int(selected_frame.shape[0] / 2)]['Audiobook_log_freq_per_million'].item()
            # Convert the median index to an integer (as iloc expects integer indices)
            new_matched_audiobook = matched_audiobook[matched_audiobook['Audiobook_log_freq_per_million'] != freq]
        
        diff_dict = get_bin_diff(CDI_bins_stats,audiobook_bins,new_matched_audiobook)
        density_diff = sum(abs(value) for value in diff_dict.values())
        
        return new_matched_audiobook, diff_dict, density_diff
    
    CDI_bins_stats = get_bin_stat(CDI_bins,matched_CDI['CHILDES_log_freq_per_million'].tolist())
    
    # compare the results each iteration
    density_diff = float('inf')
    while density_diff > threshold:
        matched_audiobook, diff_dict, density_diff = update_bins(CDI_bins_stats,audiobook_bins,matched_audiobook)
         
    return audiobook_bins, matched_audiobook 


def match_bin_prop(matched_audiobook,threshold):
    
    '''
    match distribution of the audiobook freq of machine CDI with CHILDES freq of CDI; 
    allow variations in each distr
    
    We assume Human CDI follows the equal-sized ditr here
    '''
    
    def remove_middle_rows(df, n):
        middle_index = len(df) // 2
        start_index = middle_index - n // 2
        end_index = start_index + n

        # Remove n rows from the middle of the DataFrame
        updated_df = pd.concat([df.iloc[:start_index], df.iloc[end_index:]])

        return updated_df

    # Group by the specified column
    grouped_df = matched_audiobook.groupby('group')
    # Find the minimum number of rows in the grouped DataFrames
    min_rows = int(grouped_df.size().min() * (1+threshold))

    updated_audiobook = pd.DataFrame()
    for group_name, sub_df in grouped_df:
        
        # fluctruate by prop
        
        n_rows_to_remove = sub_df.shape[0] - min_rows
        if n_rows_to_remove > 0:
            
            # Apply the custom function to each group, keeping the minimum number of rows
            updated_df = remove_middle_rows(sub_df, n_rows_to_remove)
            
        else:
            updated_df = sub_df
            
        updated_audiobook = pd.concat([updated_audiobook, updated_df])

    return updated_audiobook



def plot_density_hist(matched_CDI,freq_name,freq_type,label,alpha,mode,n_bins): 
    
    def get_first_number(group):
        return group.iloc[0] 
    
    if mode == 'given_bins':
        # first sort the dataframe by the given column
        CDI_array = np.append(matched_CDI.groupby('group')[freq_name + '_per_million'].apply(get_first_number).values
                              ,matched_CDI[freq_name + '_per_million'].tolist()[-1])
        
        data_sorted = matched_CDI[freq_name + '_per_million'].tolist()
        
        
        
    elif mode == 'equal_bins':
        
        # sort the dataframe by the required column 
        matched_CDI = matched_CDI.sort_values(by=freq_name + '_per_million')
        data = matched_CDI[freq_name + '_per_million'].tolist()
        
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
        CDI_array = np.append(bins, data_sorted[-1]+jitter)  # this is because the extreme right edge is inclusive in plt.hits
        

    # get bins based on the dataframe
    plt.hist(data_sorted,bins=CDI_array, density=True, alpha=alpha,edgecolor='black',label = label) 
    
    plt.xlabel(freq_type + '_per million')
    plt.ylabel('Density')
    
    # set the limits of the x-axis for each line
    if freq_type == 'freq':
        
        plt.xlim(0,850)
        plt.ylim(0,0.035)
            
    elif freq_type == 'log_freq':
        
        plt.xlim(-1,4)
        plt.ylim(0,1.5)

    freq_stat = get_bin_stat(CDI_array,data_sorted)
    len_stat = get_len_stat(matched_CDI,'word_len')
    # map the len columns with the freq stat
    # Concatenate along the common column
    stat = pd.concat([freq_stat.set_index('group'), len_stat.set_index('group')], axis=1, join='outer').reset_index()

    return stat
      

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

