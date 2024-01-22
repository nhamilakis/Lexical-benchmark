#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
get recep scores; annotate different groups: filter each band's words

output: concatenated scores with annotated orth form and freq bands
@author: jliu

/scratch2/jliu/STELAWord/eval/recep/wuggy/score/200h/00/Machine/speech/score_lexical_test_all_trials.csv

intersections with the tested words 
'''
import os
import sys
import pandas as pd
import argparse
import subprocess
import logging

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Get recep vocab scores')
    
    parser.add_argument('--Test_path', type=str, default = 'test_words',
                        help='Path to the test vocab and the wuggy material')
    
    parser.add_argument('--score_root', type=str, default = '/scratch2/jliu/STELAWord/eval/recep/wuggy/score/',
                        help='root Path to final scores')
    
    parser.add_argument('--modality', type=str, default = 'speech',
                        help='modalities: phoneme or speech')
    
    return parser.parse_args(argv)


score_root = '/scratch2/jliu/STELAWord/eval/recep/wuggy/score/'

# loop and find intersections

# read the test file
score = pd.read_csv('recep/score_lexical_test_all_trials.csv')
gold = pd.read_csv('stat/corpus/char/gold_test.csv',encoding='unicode_escape')
testset = pd.read_csv('/data/Lexical-benchmark/stat/freq/char/bin_range_aligned/6/1.0/matched_BE_exp.csv')



def match_orth(score_dir, gold, modality):
    
    
    def match_single_orth(score_path,gold):
        
        '''
        input:
            1.the score dataframe with each pair of word
            2.gold file with orthographic info
            
        return 
            the scoreframe with additional orth column
        '''
        
        score = pd.read_csv(score_path)
        # match the orthographic form with id
        score_grouped = score.groupby('id')
        
        score_frame = pd.DataFrame()
        for word_id,id_group in score_grouped: 
            id_group['orth'] = list(set(gold[gold['id'] == word_id]['word']))[0] 
            score_frame = pd.concat([score_frame,id_group])
        
        return score_frame
    
    score_frame_all = pd.DataFrame()
    # loop over the directory
    for dir_name in os.listdir(score_dir):
        if (dir_name != 'speech') and (dir_name != 'phoneme'):
            score_path = score_dir + '/' + dir_name + '/' + modality +'/score_lexical_test_all_trials.csv'
            
            if os.path.exists(score_path):
                score_frame = match_orth(score_path, gold)
                score_frame_all = pd.concat([score_frame_all,score_frame])
        
    return score_frame_all


def match_band(score_frame,testset):
    
    '''
    input:
        1.the scoreframe with additional orth column
        2.the reference test set with the freq bands
        
    return 
        the scoreframe with additional freq band
   
    '''
    
    score_frame_grouped = score_frame.groupby('orth')
    
    # get the number of freq bands from test set
    band_num = len(set(testset['group']))
    freq_frame = pd.DataFrame()
    
    for orth,orth_group in score_frame_grouped:
        orth_group['freq_' + str(band_num)] = list(set(testset[testset['word'] == orth]['group']))[0] 
        freq_frame = pd.concat([freq_frame,orth_group])

    return freq_frame


pair_num = 6
voice_thre = 0.875
word_thre = 0.833
modality ='phoneme'

def get_score(freq_frame,modality,voice_thre, word_thre,pair_num):
    
    '''
    get the score averaged fro each freq band
    
    input:
        the scoreframe annotated with freq band
        voice and word thre to decide whether a word is 'known' or not
        
    return 
        dataframe with avged scores by freq bands
    '''
    
    # sort by orthographic form and voice for replicability
    freq_frame = freq_frame.sort_values(by='orth')
    freq_frame = freq_frame.sort_values(by='voice')
    
    # truncate the scoreframe into target pair_num; 
    freq_frame_grouped = freq_frame.groupby('orth')    
    
    selected_score_frame = pd.DataFrame()
    
    for orth,orth_group in freq_frame_grouped:
        # only select the target pair_num
        selected_id = list(set(orth_group['id']))[:pair_num]
        selected_orth_group = orth_group[orth_group['id'].isin(selected_id)]
        selected_score_frame = pd.concat([selected_score_frame,selected_orth_group])
    
    # get the average score based on different thresholds
    selected_score_frame_grouped = selected_score_frame.groupby(selected_score_frame.columns[-1])
    
    mean_score_frame = pd.DataFrame()
    # get the average score by freq band 
    for freq, freq_group in selected_score_frame_grouped:
        
        selected_freq_group = freq_group.groupby('orth')
        # group by orthographical format
        band_score = 0
        for orth,orth_group in selected_freq_group:
        
            word_score = 0
            if modality == 'speech':
                # assign binary scores by voice_thre
                raw_score  = orth_group.groupby('id')['score'].mean().apply(lambda x: 1 if x >= voice_thre else 0)
                
            else:    
                # get the score frame by different freq bands
                raw_score  = orth_group.groupby('id')['score']
                
            word_score = raw_score.apply(lambda x: 1 if x >= word_thre else 0).mean()
            band_score += word_score
        
        # concatenate the averaged score into a large dataframe
        band_score = band_score/len(set(freq_group['orth']))
        band_frame = pd.DataFrame([freq,band_score,len(set(freq_group['orth']))]).T
        mean_score_frame = pd.concat([mean_score_frame,band_frame])
    
    mean_score_frame.rename(columns={0:'group',1:'mean_score',2:'word_num'}, inplace=True)
    
    return mean_score_frame




def main(argv):
    
    # Args parser
    args = parseArgs(argv)
    # loop over hour and chunk 
    
    # concatenate the frames from the evluation sets
    # remove nan
    gold = gold.dropna()
    
    for hour in os.listdir(args. +'/'+hour):
        
        for chunk in os.listdir():
            
            


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)


