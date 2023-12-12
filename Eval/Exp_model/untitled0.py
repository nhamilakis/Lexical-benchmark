#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample the similar amount of generations as the CHILDES distribution

@author: jliu
"""

import os
import pandas as pd
from plot_entropy_util import plot_single_para, plot_distance, match_seq,lemmatize,filter_words,get_score
import collections
import argparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Investigate CHILDES corpus')
    
    parser.add_argument('--lang', type=str, default = 'AE',
                        help='langauges to test: AE, BE or FR')
    
    parser.add_argument('--TextPath', type=str, default = 'CHILDES',
                        help='root Path to the CHILDES transcripts; one of the variables to invetigate')
    
    parser.add_argument('--OutputPath', type=str, default = 'Output',
                        help='Path to the freq output.')
    
    parser.add_argument('--condition', type=str, default = 'recep',
                        help='types of vocab: recep or exp')
    
    parser.add_argument('--hour', type=int, default = 3,
                        help='the estimated number of hours per day')
    
    parser.add_argument('--word_per_hour', type=int, default = 10000,
                        help='the estimated number of words per hour')
    
    parser.add_argument('--threshold_range', type=list, default = [10],
                        help='threshold to decide knowing a productive word or not, one of the variable to invetigate')
    
    parser.add_argument('--eval_path', type=str, default = 'Human_CDI/',
                        help='path to the evaluation material; one of the variables to invetigate')
    
    return parser.parse_args(argv)



root_path = 'generation'


def load_data(root_path, prompt_type,strategy):
    
    '''
    count all the words in the generated tokens 
    
    input: the root directory containing all the generated files adn train reference
    output: 1.the info frame with all the generarted tokens
            2.the reference frame with an additional column of the month info
            3.vocab size frame with the seq word and lemma frequencies
    '''
    
    month_dict = {'50h':[1],'100h':[1],'200h':[2,3],'400h':[4,8],'800h':[9,18],'1600h':[19,28],'3200h':[29,36]}
    
    generation_all = pd.DataFrame()
    # go over the generated files recursively
    for month in os.listdir(root_path): 
            
        for chunk in os.listdir(root_path + '/' + month): 
            
            for file in os.listdir(root_path + '/' + month + '/' +  chunk+ '/' + prompt_type + '/' + strategy):
                        
                try:
                        
                    # load decoding strategy information       
                    data = pd.read_csv(root_path + '/' + month + '/' +  chunk+ '/' + prompt_type + '/' + strategy + '/' + file)
                    data['month'] = data['month'].astype(int)
                    # sample data of the corresponding month
                    result = data.loc[(data['month'] >= month_dict[month][0]) & (data['month'] <= month_dict[month][-1])]
                    generation_all = pd.concat([generation_all,result])   
                    print('SUCCESS: ' + month+ '/' + prompt_type + '/' + strategy + '/' + file)    
                                    
                except:
                    print('FAILURE: ' + month+ '/' + prompt_type + '/' + strategy + '/' + file)
                                
     # get normalized frequency by each month
     for month in generation_all['month'].tolist():
         selected_frame = generation_all[generation_all['month']==month]
         # normalize word count in the given month
         
         seq = []
         n = 0
         while n < selected_frame.shape[0]:
             generated = selected_frame['LSTM_segmented'].tolist()[n].split(' ')
             seq.extend(generated)
             n += 1
                
         # get freq lists
         frequencyDict = collections.Counter(seq)  
         freq_lst = list(frequencyDict.values())
         word_lst = list(frequencyDict.keys())
         fre_table = pd.DataFrame([word_lst,freq_lst]).T           
         col_Names=["Word", "Freq"]
         fre_table.columns = col_Names
     
         # get the frequency that corresponds to each month
         
         seq_all.extend(seq)
     
        # get word count and lemma count frames
        seq_lst = list(set(seq_all))
        
        seq_frame = match_seq(seq_lst,frame_all)
    
    word_lst, lemma_dict = lemmatize(seq_lst)
    
    word_lst.extend(['MONTH','CHUNK','PROMPT','BEAM','TOPK','TEMP','TOPP','RANDOM','DECODING'])
    word_frame = seq_frame[word_lst]
    
    # reshape the lemma frame based onthe word_frame: basic info, lemma, total counts
    lemma_frame = seq_frame[['MONTH','CHUNK','PROMPT','BEAM','TOPK','TEMP','TOPP','RANDOM','DECODING']]
    for lemma, words in lemma_dict.items():
        # Merge columns in the list by adding their values
        lemma_frame[lemma] = word_frame[words].sum(axis=1)
        
        norm_count = fre_table[fre_table['Word']==word]['Norm_freq'].item() * 30 * word_per_hour * hour/
    return seq_frame, word_frame, lemma_frame, word_freq_frame
   






# sample data based on months


# check the overlap between wordbank and those for receptive vocab
vocab_filtered = pd.read_csv(root_path + '/vocab_filtered.csv')
vocab_unfiltered = pd.read_csv(root_path + '/vocab_unfiltered.csv')
AE_CDI = pd.read_csv(root_path + '/AE_content.csv')
BE_CDI = pd.read_csv(root_path + '/BE_content.csv')




# plot the vocab size curve based on word counts
target_frame = vocab_filtered[['word','group_3']]

# rename the column header
target_frame = target_frame.rename(columns={target_frame.columns[1]: 'freq'})


# plot the vocab size curve based on word counts
target_frame = vocab_unfiltered[['word','group']]

# rename the column header
target_frame = target_frame.rename(columns={target_frame.columns[1]: 'freq'})


def plot_curve(threshold,input_frame,target_frame,by_freq,info_lst,prompt,decoding,seq_type):
    
    '''
    input: the frame with word vocab size score

    Returns
    the learning curve of the figure

    '''
    
    sns.set_style('whitegrid')
    selected_frame, word_size  = filter_words(input_frame, target_frame,by_freq)
    
    if not by_freq:
        
        score_frame = get_score(threshold,input_frame,word_size,info_lst,prompt,decoding)
        ax = sns.lineplot(data=score_frame, x='Pseudo_month', y='Proportion of model')
    
    # get scores based on threshold based on different thresholds
    else:
        result = []
        for key, value in selected_frame.items():
           
            # selet target frame by frequency bands
            score_frame = get_score(threshold,value,word_size,info_lst,prompt,decoding)
            result.append(score_frame)
            sns.set_style('whitegrid')
            ax = sns.lineplot(data=score_frame, x='Pseudo_month', y='Proportion of model',label=key)
            
    plt.title(prompt + ' generated ' + seq_type +': ' + decoding + ', threshold: ' + str(threshold), fontsize=10)
    plt.show()

    return score_frame


threshold = 1
info_lst = ['MONTH','CHUNK','PROMPT','BEAM','TOPK','TEMP','TOPP','RANDOM']
by_freq = False
prompt = 'unprompted'
decoding = 'TOPP'
seq_type ='words'

if seq_type == 'words':
    input_frame = word_frame
    
elif seq_type == 'lemmas':
    input_frame = lemma_frame
score_frame = plot_curve(threshold,input_frame,target_frame,by_freq,info_lst,prompt,decoding,seq_type)


  
