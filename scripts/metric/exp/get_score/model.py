#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample the similar amount of generations as the CHILDES distribution

@author: jliu


"""

import os
import pandas as pd
import collections
import argparse
import enchant
from util import lemmatize

d = enchant.Dict("en_US")

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Concatenate the generated tokens by model')
    
    parser.add_argument('--root_path', type=str, default = 'AE',
                        help='root_path to the generated tokens')
    
    parser.add_argument('--prompt_type', type=str, default = 'unprompted',
                        help='prompt_type of the generation: prompted or unprompted')
    
    parser.add_argument('--OutputPath', type=str, default = 'Output',
                        help='Path to the generation stat output.')
    
    parser.add_argument('--condition', type=str, default = 'recep',
                        help='types of vocab: recep or exp')
    
    parser.add_argument('--strategy', type=str, default = 'sample_random',
                        help='the estimated number of hours per day')
    
    parser.add_argument('--word_per_hour', type=int, default = 10000,
                        help='the estimated number of words per hour')
    
    parser.add_argument('--threshold_range', type=list, default = [10],
                        help='threshold to decide knowing a productive word or not, one of the variable to invetigate')
    
    parser.add_argument('--eval_path', type=str, default = 'Human_CDI/',
                        help='path to the evaluation material; one of the variables to invetigate')
    
    return parser.parse_args(argv)




def load_data(root_path, prompt_type,strategy,word_per_sec,hour,sec_frame):
    

    '''
    count all the words in the generated tokens 
    
    input: the root directory containing all the generated files adn train reference
    output: 1.the info frame with all the generarted tokens
            2.the reference frame with an additional column of the month info
            3.vocab size frame with the seq word and lemma frequencies
    '''
    
    month_dict = {'50h':[1],'100h':[1],'200h':[2,3],'400h':[4,8],'800h':[9,18],'1600h':[19,28],'3200h':[29,36]}
    
    generation_all = pd.DataFrame()
    seq_all = []
    # go over the generated files recursively
    for month in os.listdir(root_path): 
            
        for chunk in os.listdir(root_path + '/' + month): 
            
            try:
                for file in os.listdir(root_path + '/' + month + '/' +  chunk+ '/' + prompt_type + '/' + strategy):
                        
                    # load decoding strategy information       
                    data = pd.read_csv(root_path + '/' + month + '/' +  chunk+ '/' + prompt_type + '/' + strategy + '/' + file)
                    data['month'] = data['month'].astype(int)
                    # sample data of the corresponding month
                    result = data.loc[(data['month'] >= month_dict[month][0]) & (data['month'] <= month_dict[month][-1])]
                    generation_all = pd.concat([generation_all,result])   
                    
                    n = 0
                    while n < result.shape[0]:
                        generated = result['LSTM_segmented'].tolist()[n].split(' ')
                        seq_all.extend(generated)
                        n += 1
                        
                    print('SUCCESS: ' + month+ '/' + prompt_type + '/' + strategy + '/' + file)    
                                    
            except:
                    print('FAILURE: ' + month+ '/' + prompt_type + '/' + strategy)
    
    # get the list of all the words
    seq_lst_temp = list(set(seq_all))
    seq_lst = [element for element in seq_lst_temp if element.strip() != '']   
    cleaned_frame = pd.DataFrame(seq_lst)
    
    for i in list(set(generation_all['month'].tolist())):
        cleaned_frame[i] = [0] * len(seq_lst)
    
    cleaned_frame.set_index(0, inplace=True)
    
    # Fill the DataFrame with zeros for the specified number of rows
    for month in list(set(generation_all['month'].tolist())):
        
        try:
            sec_per_hour = sec_frame[sec_frame['month']==month]['sec_per_hour'].item()
                    
            selected_frame = generation_all[generation_all['month']==month]
            seq = []
            n = 0
            while n < selected_frame.shape[0]:
                generated = selected_frame['LSTM_segmented'].tolist()[n].split(' ')
                seq.extend(generated)
                n += 1
            # get the frequency that corresponds to each month
            filtered_seq = [element for element in seq if element.strip() != '']       
            # get freq lists
            frequencyDict = collections.Counter(filtered_seq)  
            
            for word, freq in frequencyDict.items():
                    cleaned_frame.loc[word,month] = freq/len(filtered_seq)* 30 * word_per_sec * hour * sec_per_hour
                    n += 1
        except:
            print(month)
    
    # get cumulative frequency
    seq_frame = cleaned_frame.cumsum(axis=1)
    
    # get the lemam frame and then match the results
    lemma_dict = lemmatize(seq_lst)
    
    lemma_frame = pd.DataFrame()
    for lemma, words in lemma_dict.items():
        # Merge columns in the list by adding their values
        # Sum all rows to get one row with summed values
        df = seq_frame.loc[words]
        summed_row = df.sum(axis=0)
        # Convert the resulting Series back to a DataFrame with one row
        summed_df = pd.DataFrame([summed_row], columns=summed_row.index)
        lemma_frame = pd.concat([lemma_frame,summed_df])
    
    lemma_frame.index = list(lemma_dict.keys())

    return seq_frame, lemma_frame


root_path = 'generation'
strategy = 'sample_random'

vocab_type = 'word'   

condition = 'Expressive'
sec_frame = pd.read_csv('/data/Lexical-benchmark/vocal_month.csv')

# load data
seq_frame_prompted,lemma_frame_prompted = load_data(root_path, 'prompted',strategy,word_per_sec,hour,sec_frame)
seq_frame_unprompted,lemma_frame_unprompted = load_data(root_path, 'unprompted',strategy,word_per_sec,hour,sec_frame)


save_root = '/data/Lexical-benchmark/exp/reference/generation_stat/'
seq_frame_prompted.to_csv(save_root + 'prompted.csv')     
seq_frame_unprompted.to_csv(save_root + 'unprompted.csv')  

