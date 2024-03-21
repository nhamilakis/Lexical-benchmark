#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construct monthly correspondent CHILDES transcript

returns a concatenated csv file with all the info in the CHILDES

@author: jliu
"""
import numpy as np
import os
from Preprocess_util import count_token
import pandas as pd
from tqdm import tqdm 
import enchant

d = enchant.Dict("en_US")


def clean_all(extracted_dir):
    
    log_set = set()
    transcript = pd.DataFrame()
    for lang in os.listdir(extracted_dir):
        # dive into the corpus
        lang_folder = os.path.join(extracted_dir, lang)
      
        for corpus in os.listdir(lang_folder):   
            # loop over the folders until finding the target transcript
            corpus_folder = os.path.join(lang_folder, corpus)
            
            for item in tqdm(os.listdir(corpus_folder)):  
                # check existence of .cha file
                cha_path = os.path.join(corpus_folder, item)
                if not os.path.isdir(cha_path):
                    try:
                        trans_frame = pd.read_csv(cha_path)
                        trans_frame['num_tokens'] = trans_frame['content'].apply(count_token)
                        trans_frame['path'] = cha_path
                        trans_frame['filename'] = item
                        transcript = pd.concat([transcript,trans_frame])
                        
                    except:
                        print(cha_path)
                        log_set.add(cha_path)
                        
                else:
                    # loop over the folder
                    for item in tqdm(os.listdir(cha_path)):  
                        # check existence of .cha file
                        cha_folder = os.path.join(cha_path, item)
                        
                        if not os.path.isdir(cha_folder):
                            try:
                                trans_frame = pd.read_csv(cha_folder)
                                trans_frame['num_tokens'] = trans_frame['content'].apply(count_token)
                                trans_frame['path'] = cha_folder
                                trans_frame['filename'] = item
                                transcript = pd.concat([transcript,trans_frame])
                            except:
                                print(cha_folder)
                                log_set.add(cha_folder)
                        else:
                            
                            for item in tqdm(os.listdir(cha_folder)):  
                                cha_dir = os.path.join(cha_folder, item)
                                if not os.path.isdir(cha_dir):
                                    try:
                                        trans_frame = pd.read_csv(cha_dir)
                                        trans_frame['num_tokens'] = trans_frame['content'].apply(count_token)
                                        trans_frame['path'] = cha_dir
                                        trans_frame['filename'] = item
                                        transcript = pd.concat([transcript,trans_frame])
                                    
                                    except:
                                        print(cha_dir)
                                        log_set.add(cha_dir)       
                                    
                                    

trans_path = '/data/Machine_CDI/Lexical-benchmark_data/exp/CHILDES/cleaned_transcript/all_trans.csv'
vocal_frame_path = '/data/Machine_CDI/Lexical-benchmark_data/exp/vocal_month.csv'
# get the stat of the cleaned CHILDES data; save in the vocal_month.csv
def get_stat(trans_path, vocal_frame_path):
    
    
    def get_speaker_stat(trans_child,speaker):
        
        def seg_string(text):
            return text.split(' ')
        
        child_frame = pd.DataFrame()
        trans_child_grouped = trans_child.groupby('month')
        for month, trans_child_group in trans_child_grouped:
        
            # get token type
            token_lst = [x for xs in trans_child_group['content'].apply(seg_string).tolist() for x in xs]
            token_type_num = len(set(token_lst))
            # decide whehther it is a word
            word_lst = []
            for token in list(set(token_lst)):
                if d.check(token):
                    word_lst.append(token)
                    
            # get the number of words in the token list
            # Initialize a dictionary to store counts
            count_dict = {}
            # Count occurrences of elements in list B
            for element in token_lst:
                if element in word_lst:
                    if element in count_dict:
                        count_dict[element] += 1
                    else:
                        count_dict[element] = 1
            word_num = sum(count_dict.values())
            word_type_num = len(set(word_lst))
            month_frame = pd.DataFrame([[month, trans_child_group.shape[0],trans_child_group['num_tokens'].sum()
                                         ,token_type_num,word_num,word_type_num]])
            child_frame = pd.concat([child_frame,month_frame])
        child_frame.columns = ['month', speaker + '_num_utt',speaker + '_num_tokens',speaker + '_num_token_type',
                               speaker + '_num_words', speaker + '_num_word_type']
        return child_frame
    
    
    transcript = pd.read_csv(trans_path)
    vocal_frame = pd.read_csv(vocal_frame_path,index_col = 'month')
    # go over the frames
    trans_child = transcript[transcript['speaker'] == 'CHI']
    stat_child = get_speaker_stat(trans_child,'child')
    # aggregate by month: get utterance and token number 
    trans_adult = transcript[transcript['speaker'] != 'CHI']
    stat_adult = get_speaker_stat(trans_adult,'adult')
    stat_child_selected = stat_child[stat_child['month'].isin(stat_adult['month'])]
    stat_child_selected = stat_child_selected.drop(columns=['month'])
    # concatnate two df
    stat_all = pd.concat([stat_child_selected, stat_adult], axis=1).reset_index(drop=True)
    
    # append NA to vocal frame
    additional_df = pd.DataFrame(np.nan, index=np.arange(stat_all.shape[0] - vocal_frame.shape[0]), columns=vocal_frame.columns)
    result_vocal = pd.concat([vocal_frame, additional_df], axis=0, ignore_index=True)
    stat_final = pd.concat([result_vocal, stat_all], axis=1)
    
    # Extract the column 'C'
    month_column= stat_final.pop('month')
    # Insert the column 'C' at the most left
    stat_final.insert(0, 'month', month_column)
    
    # append adult estimation assume 3h per day , 10,000 words per hour in avg
    stat_final['adult_month_est'] = stat_final['month'] * 30 * 10000 * 3
    stat_final.to_csv(vocal_frame_path)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
