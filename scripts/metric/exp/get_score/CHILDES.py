#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply thresholds on CHILDES scripts or model generations

@author: jliu
"""
import os
from util import load_transcript, get_freq, count_by_month, get_score
import pandas as pd
import argparse
import sys
import numpy as np


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Apply thresholds on generation/production')
    
    parser.add_argument('--lang', type=str, default = 'BE',
                        help='languages to test: AE or BE')
    
    parser.add_argument('--eval_type', type=str, default = 'CDI',
                        help='languages to test: CDI or model')
    
    parser.add_argument('--eval_condition', type=str, default = 'exp',
                        help='which type of words to evaluate; recep or exp')
    
    parser.add_argument('--TextPath', type=str, default = 'CHILDES',
                        help='root Path to the CHILDES transcripts')
    
    parser.add_argument('--OutputPath', type=str, default = 'Output',
                        help='Path to the freq output.')
    
    parser.add_argument('--input_condition', type=str, default = 'exp',
                        help='types of vocab: recep or exp')
    
    parser.add_argument('--hour', type=dict, default = 10,
                        help='the estimated number of waking hours per day; data from Alex(2023)')
    
    parser.add_argument('--word_per_sec', type=int, default = 3,
                        help='the estimated number of words per second')
    
    parser.add_argument('--sec_frame_path', type=str, default = 'vocal_month.csv',
                        help='the estmated vocalization seconds per hour by month')
    
    parser.add_argument('--threshold_range', type=list, default = [50,100,200,300],
                        help='threshold to decide knowing a productive word or not, one of the variable to invetigate')
    
    parser.add_argument('--eval_path', type=str, default = 'Human_eval/',
                        help='path to the evaluation material; one of the variables to invetigate')
    
    return parser.parse_args(argv)



def load_data(TextPath,OutputPath,lang,input_condition):
    
    '''
    get word counts from the cleaned transcripts
    input: text rootpath with CHILDES transcripts
    output: freq_frame, score_frame
    '''
    
    # load and clean transcripts: get word info in each seperate transcript
    if os.path.exists(OutputPath + '/stat_per_file' +'.csv'):     
        print('The transcripts have already been cleaned! Skip')
        file_stat_sorted = pd.read_csv(OutputPath + '/stat_per_file' +'.csv')
        
    # check whether the cleaned transcripts exist
    else:
        print('Start cleaning files')
        file_stat = pd.DataFrame()
        for lang in os.listdir(TextPath):  
            output_dir =  OutputPath + '/' + lang
            for file in os.listdir(TextPath + '/' + lang): 
                try: 
                    file_frame = load_transcript(TextPath,output_dir,file,lang,input_condition)
                    file_stat = pd.concat([file_stat,file_frame])
                    
                except:
                    print(file)
                    
        file_stat_sorted = file_stat.sort_values('month')    
        file_stat_sorted.to_csv(OutputPath + '/stat_per_file' +'.csv')                
        print('Finished cleaning files')
    
    # concatenate word info in each month
    month_stat = count_by_month(OutputPath,file_stat_sorted)
    
    return month_stat



def count_words(OutputPath,group_stat,eval_path,hour,word_per_sec,eval_type,lang,eval_condition,sec_frame):
    
    '''
    count words of the given list
    '''
    
    eval_dir = eval_path + eval_type + '/' + lang + '/' + eval_condition
           
    for file in os.listdir(eval_dir):
        eval_frame = pd.read_csv(eval_dir + '/' + file)
        eval_lst = eval_frame['word'].tolist()
        
    freq_frame = pd.DataFrame()
    freq_frame['word'] = eval_lst
    freq_frame['group_original'] = eval_frame['group_original'].tolist()
    
    # loop each month
    for file in set(group_stat['end_month'].tolist()):
        
        # get word freq list for each file
        text_file = 'transcript_' + str(file)
        file_path = OutputPath + '/Transcript_by_month/' + text_file + '.txt'  
        
        with open(file_path, encoding="utf8") as f:
            sent_lst = f.readlines()
        word_lst = []    
        for sent in sent_lst:
            # remove the beginning and ending space
            words = sent.split(' ')
            
            for word in words:
                cleaned_word = word.strip()
                if len(cleaned_word) > 0:  
                    word_lst.append(cleaned_word)
        # save the overall freq dataframe for further use
        fre_table = get_freq(word_lst)

        freq_lst = []
        for word in eval_lst:
            try: 
                # recover to the actual count based on Alex's paper
                sec_per_hour = sec_frame[sec_frame['month']==file]['sec_per_hour'].item()
                norm_count = fre_table[fre_table['Word']==word]['Norm_freq'].item() * 30 * word_per_sec * hour * sec_per_hour
                
            except:
                norm_count = 0
            freq_lst.append(norm_count)
        freq_frame[file] = freq_lst
        
    # sort the target frameRecep vocab
    
    # we use cum freq as the threshold for the word
    sel_frame = freq_frame.iloc[:,2:]
    columns = freq_frame.columns[2:]
    sel_frame = sel_frame.cumsum(axis=1)
            
    for col in columns.tolist():
        freq_frame[col] = sel_frame[col]
    
    freq_frame.to_csv(OutputPath + '/selected_freq.csv')
    
    return freq_frame



def main(argv):
    
    # Args parser
    args = parseArgs(argv)
    
    TextPath = args.TextPath
    eval_condition = args.eval_condition
    input_condition = args.input_condition
    lang = args.lang
    eval_type = args.eval_type
    OutputPath = args.OutputPath + '/' + eval_type + '/' + lang + '/' + eval_condition
    eval_path = args.eval_path
    hour = args.hour
    threshold_range = args.threshold_range
    word_per_sec = args.word_per_sec
    sec_frame = pd.read_csv(args.sec_frame_path)
        
    # step 1: load data and count words
    month_stat = load_data(TextPath,OutputPath,lang,input_condition)
    freq_frame = count_words(OutputPath,month_stat,eval_path,hour,word_per_sec,eval_type,lang,eval_condition,sec_frame)



   

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    


