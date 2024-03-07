#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics for human production/gerneration
"""
import os
#from util import load_transcript, get_freq, count_by_month, get_score
import pandas as pd
import argparse
import sys
from collections import Counter

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Investigate CHILDES corpus')
    
    parser.add_argument('--lang', type=str, default = 'BE',
                        help='languages to test: AE, BE or FR')
    
    parser.add_argument('--eval_type', type=str, default = 'CDI',
                        help='languages to test: human or model')
    
    parser.add_argument('--eval_condition', type=str, default = 'exp',
                        help='which type of words to evaluate; recep or exp')
    
    parser.add_argument('--TextPath', type=str, default = '/data/Machine_CDI/Lexical-benchmark_data/exp/CHILDES',
                        help='root Path to the CHILDES transcripts; one of the variables to invetigate')
    
    parser.add_argument('--OutputPath', type=str, default = '/data/Machine_CDI/Lexical-benchmark_output/Final_scores/Model_eval/exp/',
                        help='Path to the freq output.')
    
    parser.add_argument('--input_condition', type=str, default = 'recep',
                        help='recep for parent production or exp for children production')
    
    parser.add_argument('--sec_frame_path', type=str, default = '/data/Machine_CDI/Lexical-benchmark_data/exp/vocal_month.csv',
                        help='the estmated vocalization seconds per hour by month')
    
    parser.add_argument('--threshold_range', type=list, default = [20,60,200,600],
                        help='threshold to decide knowing a productive word or not, one of the variable to invetigate')
    
    parser.add_argument('--eval_path', type=str, default = '/data/Machine_CDI/Lexical-benchmark_output/test_set/matched_set/char/bin_range_aligned/6_audiobook_aligned/',
                        help='path to the evaluation material; one of the variables to invetigate')
    
    parser.add_argument('--mode', type=str, default = 'constant',
                        help='constant or varying')
    
    return parser.parse_args(argv)

TextPath = '/data/Machine_CDI/Lexical-benchmark_data/test_set/freq_corpus/char/CHILDES_trans.csv'
age_range = [0,36]
eval_type = 'child'
sec_frame_path = '/data/Machine_CDI/Lexical-benchmark_data/exp/vocal_month.csv'
df = trans_speaker
content_header = 'content'
eval_path = '/data/Machine_CDI/Lexical-benchmark_output/test_set/matched_set/char/bin_range_aligned/6_audiobook_aligned/'
OutputPath =  '/data/Machine_CDI/Lexical-benchmark_output/Final_scores/Model_eval/exp/'

def count_by_month(df,content_header,mode, sec_frame):
    '''
    count words by month 
    return raw count, adjusted count and accum count
    '''
    # Initialize an empty dictionary to store word counts for each month
    word_counts_by_month = {}
    
    # Iterate over each row of the DataFrame
    for index, row in df.iterrows():
        # Split the string containing words by whitespace
        words = row[content_header].split()
        # Get the month of the current row
        month = row['month']
        # Initialize Counter for the current month if not exists
        if month not in word_counts_by_month:
            word_counts_by_month[month] = Counter()
        # Update word counts for the current month
        word_counts_by_month[month].update(words)
    
    # Create a set of all unique words
    all_words = set(word for words_counter in word_counts_by_month.values() for word in words_counter)
    # Create a DataFrame from the word counts dictionary with all words as index
    word_month_counts = pd.DataFrame({month: {word: counter[word] for word in all_words} for month, counter in word_counts_by_month.items()}).fillna(0).astype(int)
    # Sort month numbers in ascending order
    sorted_month_numbers = sorted(word_month_counts.columns)
    # Sort DataFrame columns in ascending order
    raw_counts = word_month_counts.reindex(sorted_month_numbers, axis=1)
    # get the adjusted count
    adjusted_counts = pd.DataFrame(index=raw_counts.index.tolist())
    n= 0
    while n < sec_frame.shape[0]:
        month = sec_frame['month'].tolist()[n]
        if mode == 'varying':
            adjusted_counts[month] = raw_counts[month]/sec_frame['num_tokens'].tolist()[n] * sec_frame['month_est'].tolist()[n]
        else:
            # assume 3h inputs and 10000 words per hour
            adjusted_counts[month] = raw_counts[month]/sec_frame['num_tokens'].tolist()[n] * 30 * 10000 * 3
        n += 1
    
    # get cum count 
    cum_counts = raw_counts.cumsum(axis=1)
    adjusted_counts_cum = adjusted_counts.cumsum(axis=1)
    return word_month_counts, cum_counts, adjusted_counts_cum






def load_data(TextPath:str,sec_frame_path:str,OutputPath,age_range:list,mode:str,eval_type:str):
    
    '''
    para: 
        -TextPath,path of the .csv files with all one column of the scripts
        -eval_type: which system to evaluate(adult,kids or model's generation)
        -age_range: min and max age to inspect 
    return: freq_frame, score_frame
    '''
    if os.path.exists(TextPath):
        # load and clean transcripts: get word info in each seperate transcript
        transcript = pd.read_csv(TextPath)
        
        # select based on age span
        transcript['month'] = transcript['month'].astype(int)
        trans_age = transcript[(transcript['month'] >= age_range[1]) & (transcript['month'] <= age_range[1])]
        
        # remove condition later as it's not necessary
        if eval_type == 'child' or eval_type == 'generation':
            trans_speaker = trans_age[trans_age['speaker'] == 'CHI']
            
        elif eval_type == 'adult':
            trans_speaker = trans_age[trans_age['speaker'] != 'CHI']
        
        # load vocal frame
        sec_frame = pd.read_csv(sec_frame_path)
        # append month_distr
        token_num_dict = trans_speaker.groupby('month')['num_tokens'].sum().to_dict()
        num_token_lst = []
        n = 0
        while n < sec_frame.shape[0]:
            num_token_lst.append(token_num_dict[sec_frame['month'].tolist()[n]])
            n += 1
        
        sec_frame['num_tokens'] = num_token_lst
        
        # count token numbers: return both raw counts and adjusted counts
        raw_counts, cum_counts, cum_adjusted_counts = count_by_month(trans_speaker,'content','varying',sec_frame)
        # output the results
        raw_counts.to_csv(OutputPath+'raw_child.csv')
        cum_counts.to_csv(OutputPath+'cum_child.csv')
        cum_adjusted_counts.to_csv(OutputPath+'cum_adjusted_child.csv')

        return raw_counts, cum_counts, cum_adjusted_counts
        
    else:
        print('No cleaned transcripts. Please do the preprocessing first ')
        
lang = 'BE'
seq_frame = cum_adjusted_counts
target_frame = test_set

def select_words(eval_path, OutputPath,lang,eval_type):
    
    def get_score(target_frame,seq_frame):

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
        return selected_frame
    
    cum_adjusted_counts = pd.read_csv(OutputPath +'cum_adjusted_'+eval_type+'.csv', index_col=0)
    eval_frame = eval_path + 'human_' + lang + '_exp.csv'
    test_set = pd.read_csv(eval_frame)
    test_frame =  get_score(test_set,cum_adjusted_counts)
    
    if not os.path.exists(OutputPath + lang):  
        os.mkdir(OutputPath + lang)
    test_frame.to_csv(OutputPath + lang + '/' + eval_type + '.csv')
    return test_frame
    
    
def main(argv):
    
    # Args parser
    args = parseArgs(argv)
    
    TextPath = args.TextPath
    sec_frame_path = args.sec_frame_path
    mode = args.mode
    OutputPath = args.OutputPath
    age_range = args.age_range
    lang = args.lang
    eval_type = args.eval_type
    OutputPath = args.OutputPath
    
    
    # check if the fig exists
    if not os.path.exists(OutputPath + '/cum_adjusted_' + eval_type + '.csv'):   
        # step 1: load data and count words
        raw_counts, cum_counts, cum_adjusted_counts = load_data(TextPath,sec_frame_path,OutputPath,age_range,mode,eval_type)
                           
    # step 2: select words from the adjusted counts
    select_words(eval_path, OutputPath,lang,eval_type)

   

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    
