#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 22:36:20 2023

@author: jliu
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def get_score(target_frame,seq_frame,threshold):
    
    
    '''
    get the weighted score based on different frequency range
    
    '''
    
    overlapping_words = [col for col in target_frame['word'].tolist() if col in seq_frame.index.tolist()]
    # get the subdataframe
    selected_frame = seq_frame.loc[overlapping_words]
    
    # use weighted class to decrase the effect of the unbalanced dataset
    
    extra_word_lst = [element for element in target_frame['word'].tolist() if element not in overlapping_words]
    
    for word in extra_word_lst:
        selected_frame.loc[word] = [0] * selected_frame.shape[1]
        
    # get the score based on theshold
    score_frame_all = selected_frame.applymap(lambda x: 1 if x >= threshold else 0)
    
    avg_values = score_frame_all.mean()
    
    return score_frame_all, avg_values



def load_CDI(human_result):
    
    '''
    load human CDI data for figure plot
    '''
    
    size_lst = []
    month_lst = []
    
    n = 0
    while n < human_result.shape[0]:
        size_lst.append(human_result.iloc[n])
        headers_list = human_result.columns.tolist()
        month_lst.append(headers_list)
        n += 1
    
    size_lst_final = [item for sublist in size_lst for item in sublist]
    month_lst_final = [item for sublist in month_lst for item in sublist]
    month_lst_transformed = []
    for month in month_lst_final:
        month_lst_transformed.append(int(month))
    # convert into dataframe
    data_frame = pd.DataFrame([month_lst_transformed,size_lst_final]).T
    data_frame.rename(columns={0:'month',1:'Proportion of acquired words'}, inplace=True)
    data_frame_final = data_frame.dropna(axis=0)
    return data_frame_final



def load_accum(accum_all,accum_threshold):
    
    df = accum_all[accum_all['threshold']==accum_threshold]
    accum_result = df.groupby(['month'])['Lexical score'].mean().reset_index()
    
    return accum_result


def load_exp(seq_frame_unprompted,target_frame,by_freq,exp_threshold):
    
    
    seq_frame_unprompted = seq_frame_unprompted.rename_axis('Index')
    
    if not by_freq:
        avg_values_unprompted_lst = []
        # decompose the results by freq groups
        for freq in set(list(target_frame['group_original'].tolist())):
            word_group = target_frame[target_frame['group_original']==freq]
            # take the weighted average by the proportion of different frequency types
            score_frame_unprompted, avg_values_unprompted = get_score(word_group,seq_frame_unprompted,exp_threshold)
            
            avg_values_unprompted_lst.append(avg_values_unprompted.values)
            
        avg_unprompted = (avg_values_unprompted_lst[0] + avg_values_unprompted_lst[1]) / 2    
    
    # or we just read single subdataframe
    else:
        score_frame_unprompted, avg_unprompted = get_score(target_frame,seq_frame_unprompted,exp_threshold)
    
    return score_frame_unprompted, avg_unprompted 




def get_score_CHILDES(freq_frame,threshold):
    
    '''
    get scores of the target words in Wordbank 
    input: word counts in each chunk/month and the stat of the number of true words as well as the true data proportion
    output: a dataframe with each word count  

    we have the weighed score in case the influence of different proportions of frequency bands      
    '''
     
    
    freq_frame = freq_frame.drop(columns=['word', 'group_original'])
    
    # get each chunk's scores based on the threshold
    columns = freq_frame.columns
      
    for col in columns.tolist():
        freq_frame.loc[col] = [0] * freq_frame.shape[1]
        
    # get the score based on theshold
    score_frame = freq_frame.applymap(lambda x: 1 if x >= threshold else 0)
    
    avg_values = score_frame.mean()
    return score_frame,avg_values