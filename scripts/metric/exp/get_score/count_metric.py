#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics for human production/gerneration
"""
import os
from metric_util import count_by_month
import pandas as pd
import argparse
import sys

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Count words by month')
    
    parser.add_argument('--lang', type=str, default = 'BE',
                        help='languages to test: AE, BE or FR')
    
    parser.add_argument('--eval_type_lst', default = ['child','unprompted_0.3','unprompted_0.6','unprompted_1.0','unprompted_1.5'],
                        help='system to test: child, adult, train or generation')
    
    parser.add_argument('--TextPath', type=str, default = '/data/Machine_CDI/Lexical-benchmark_output/Final_scores/Model_eval/exp/generation.csv',
                        help='root Path to the CHILDES transcripts; one of the variables to invetigate')
    
    parser.add_argument('--OutputPath', type=str, default = '/data/Machine_CDI/Lexical-benchmark_output/Final_scores/Model_eval/exp/count/',
                        help='Path to the freq output.')
    
    parser.add_argument('--sec_frame_path', type=str, default = '/data/Machine_CDI/Lexical-benchmark_data/exp/vocal_month.csv',
                        help='the estmated vocalization seconds per hour by month')
    
    parser.add_argument('--eval_path', type=str, default = '/data/Machine_CDI/Lexical-benchmark_output/test_set/matched_set/char/bin_range_aligned/12_audiobook_aligned/',
                        help='path to the evaluation material; one of the variables to invetigate')
    
    parser.add_argument('--mode', type=str, default = 'varying',
                        help='constant or varying')
    
    parser.add_argument('--age_range', default = [6,36],
                        help='constant or varying')
    
    return parser.parse_args(argv)

'''
TextPath = '/data/Machine_CDI/Lexical-benchmark_output/Final_scores/Model_eval/exp/generation.csv'
eval_type = 'unprompted_0.3'
OutputPath = '/data/Machine_CDI/Lexical-benchmark_output/Final_scores/Model_eval/exp/count/'
lang = 'AE'
eval_path = '/data/Machine_CDI/Lexical-benchmark_output/test_set/matched_set/char/bin_range_aligned/6_audiobook_aligned/'
age_range = [6,36]
sec_frame_path = '/data/Machine_CDI/Lexical-benchmark_data/exp/vocal_month.csv'
mode = 'varying'
'''


def load_data(TextPath:str,sec_frame_path:str,OutputPath,age_range:list,mode:str,eval_type:str,lang):
    
    '''
    para: 
        -TextPath,path of the .csv files with all one column of the scripts
        -eval_type: which system to evaluate(adult,kids or model's generation)
        -age_range: min and max age to inspect 
    return: freq_frame, score_frame (for oov analysis)
    '''
    if os.path.exists(TextPath):
        
        # load and clean transcripts: get word info in each seperate transcript
        transcript = pd.read_csv(TextPath)
        
        if eval_type == 'child' or eval_type.startswith('unprompted'):
            
            # select lang
            trans_speaker = transcript[transcript['lang'] == lang]
            
            
        elif eval_type == 'adult':      #this is only for oov analysis
            
            # select based on age span
            transcript['month'] = transcript['month'].astype(int)
            trans_age = transcript[(transcript['month'] >= age_range[0]) & (transcript['month'] <= age_range[1])]
            trans_speaker = trans_age[trans_age['speaker'] != 'CHI']
            
        elif eval_type == 'adult_concat':
            
            # select based on speaker
            trans_speaker = transcript[transcript['speaker'] != 'CHI']
            # duplicate and re-segment the transcript without earlier month inflation
            est_token  = 10000 * 3 * 30 * age_range[1]
            # concatenate the utterances 
            n_times = est_token//trans_speaker['num_tokens'].sum() + 1
            utt = pd.DataFrame()
            n = 0
            while n < n_times:
                utt = pd.concat([utt,trans_speaker])
                n += 1
            
            utt['token_sum'] = utt['num_tokens'].cumsum()
            trans_speaker = pd.DataFrame()
            n = 0
            while n < age_range[1]:
                # select the subdataframes iteratively
                selected_frame = utt[(utt['token_sum'] >= 10000 * 3 * 30 * n) & (utt['token_sum'] < 10000 * 3 * 30 * (n + 1))]
                selected_frame['month'] = n + 1
                trans_speaker = pd.concat([trans_speaker,selected_frame])
                n += 1
            
        
        # load vocal frame
        sec_frame = pd.read_csv(sec_frame_path)
        # select the month range in the vocal frame
        sec_frame = sec_frame[(sec_frame['month'] >= age_range[0]) & (sec_frame['month'] <= age_range[1])]
        # append month_distr
        token_num_dict = trans_speaker.groupby('month')['num_tokens'].sum().to_dict()
        
        num_token_lst = []
        # select part of the dataframe
        sec_frame = sec_frame[sec_frame['month'].isin(token_num_dict.keys())]
         
        n = 0
        while n < sec_frame.shape[0]:
            num_token_lst.append(token_num_dict[sec_frame['month'].tolist()[n]])
            n += 1
        
        sec_frame['num_tokens'] = num_token_lst
        sec_frame.to_csv('/data/Machine_CDI/Lexical-benchmark_data/exp/vocal_month_' + lang + '.csv')
        
        
        # count token numbers: return both raw counts and adjusted counts
        if not eval_type.startswith('unprompted'):
            content_header = 'content'
            
        if eval_type.startswith('unprompted'):
            content_header = eval_type 
            
        raw_counts, cum_counts, cum_adjusted_counts = count_by_month(trans_speaker,content_header,mode,sec_frame,eval_type)
        # output the results
        if not eval_type == 'model':
            suffix = lang + '_' + eval_type + str(age_range[0]) + '_' + str(age_range[1]) + '.csv'
        else:
            suffix = eval_type + str(age_range[0]) + '_' + str(age_range[1]) + '.csv'
        raw_counts.to_csv(OutputPath+'raw_' + suffix)
        cum_counts.to_csv(OutputPath+'cum_' + suffix)
        cum_adjusted_counts.to_csv(OutputPath+'cum_adjusted_' + suffix)
        
    else:
        print('No cleaned transcripts. Please do the preprocessing first ')





def select_words(eval_path, OutputPath,lang,eval_type,age_range):
    
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
    
      
    suffix = lang + '_' + eval_type + str(age_range[0]) + '_' + str(age_range[1]) + '.csv'
    if eval_type == 'adult_concat' :
        cum_adjusted_counts = pd.read_csv(OutputPath +'cum_'+ eval_type + '_' + lang + '_' + str(age_range[0]) + '_' + str(age_range[1]) +'.csv', index_col=0)
    
    elif eval_type == 'model' :
        cum_adjusted_counts = pd.read_csv(OutputPath +'cum_'+ eval_type +'.csv', index_col=0)
    
    else:
        cum_adjusted_counts = pd.read_csv(OutputPath +'cum_adjusted_'+ suffix, index_col=0)
                    
    if eval_type == 'child' or eval_type.startswith('adult'):
        test_type = 'human_'
    else:
        test_type = 'machine_'
        
    eval_frame = eval_path + test_type + lang + '_exp.csv'
    test_set = pd.read_csv(eval_frame)
    test_frame =  get_score(test_set,cum_adjusted_counts)
    
    if not os.path.exists(OutputPath + lang):  
        os.mkdir(OutputPath + lang)
    test_frame.to_csv(OutputPath + lang + '/' + eval_type + '_' + str(age_range[0]) + '_' + str(age_range[1])+ '.csv')
    
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
    eval_type_lst = args.eval_type_lst
    OutputPath = args.OutputPath
    eval_path = args.eval_path
    
    for eval_type in eval_type_lst:
        # check if the fig exists
        if not os.path.exists(OutputPath+'cum_adjusted_' + eval_type + str(age_range[0]) + '_' + str(age_range[1]) +'.csv'):   
            # step 1: load data and count words
            print('Counting words from {}'.format(eval_type))
            load_data(TextPath,sec_frame_path,OutputPath,age_range,mode,eval_type,lang)
        else:
            print('{} word freq has been created'.format(eval_type))                 
        # step 2: select words from the adjusted counts
        select_words(eval_path, OutputPath,lang,eval_type,age_range)

   

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    

