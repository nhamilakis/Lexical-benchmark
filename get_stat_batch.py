#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get statistics of the results recursively

@author: jliu
"""
import os
import subprocess
import pandas as pd
from stat_util import get_overlapping
from functools import reduce

def match_stat(freq_path):
    
    # also get the range of different sets' range
    
    def get_inter(nested_list):
        # Find intersection using reduce and set.intersection
        intersection = reduce(set.intersection, map(set, nested_list))
        intersection_list = list(intersection)
        return len(intersection_list)
        
    # get length and overlapping
    recep_lst = []
    exp_lst = []
    matched_lst = []
    CDI_lst = []
    all_lst = []
    len_dict = {}
    word_lists_dict = {}
    for file in os.listdir(freq_path):
        if file.endswith('.csv'):
            
            test = pd.read_csv(freq_path + '/' + file)
            len_dict[file] = test.shape[0]   
            word_lists_dict[file] = test['word'].tolist()
            all_lst.append(test['word'].tolist())
            
            if file.split('.')[0].split('_')[-1] == 'recep':
                recep_lst.append(test['word'].tolist())
            elif file.split('.')[0].split('_')[-1] == 'exp':
                exp_lst.append(test['word'].tolist())
                
            if file.split('.')[0].split('_')[0] == 'CDI':
                CDI_lst.append(test['word'].tolist())
            if file.split('.')[0].split('_')[0] == 'matched':
                matched_lst.append(test['word'].tolist())
          
    len_dict['recep'] = get_inter(recep_lst)
    len_dict['exp'] = get_inter(exp_lst)
    len_dict['matched'] = get_inter(matched_lst)
    len_dict['CDI'] = get_inter(CDI_lst)
    len_dict['all'] = get_inter(all_lst)
    # get overlapping of the similar type of datasets
    len_frame = pd.DataFrame.from_dict(len_dict, orient='index')
    
    matched_stat = get_overlapping(word_lists_dict)
    return len_frame,matched_stat


def run_command(command):
    subprocess.call(command, shell=True)
    
    

# download one file to test
stat_command_temp = 'python get_stat.py --lang {lang_temp} --eval_condition {eval_condition_temp} \
                --word_format {word_format_temp} --num_bins {num_bins_temp} --freq_type {freq_type_temp} \
                --match_mode {match_mode_temp} --threshold {threshold_temp}'

                
lang_lst = ['AE','BE']

eval_condition_lst = ['exp','recep']
#word_format_lst = ['char','phon']
word_format_lst = ['char']
#match_mode_lst = ['range_aligned','bin_range_aligned']

match_mode_lst = ['distr_aligned']
num_bins_lst = [2,3,4,5]
freq_type_lst = ['freq','log_freq']

threshold_lst = [0.1]




for lang in lang_lst:
    
    for match_mode in match_mode_lst:
    
        for eval_condition in eval_condition_lst:
            
            for word_format in word_format_lst:
                
                for num_bins in num_bins_lst:
                    
                    for freq_type in freq_type_lst:
                        
                        for threshold in threshold_lst:
                            
                            
                            stat_command = stat_command_temp.format(lang_temp = lang, eval_condition_temp = eval_condition
                                             ,word_format_temp= word_format, num_bins_temp = num_bins, freq_type_temp = freq_type
                                             , match_mode_temp = match_mode, threshold_temp = threshold)
                            
                            run_command(stat_command)
                            
                            
                            freq_path = '/data/Lexical-benchmark/stat/freq/char/' + match_mode + '/' + str(num_bins) + '/' + str(threshold)
                            try:
                                len_frame,matched_stat = match_stat(freq_path)
                                # output the results
                                stat_path = freq_path + '/stat/'
                                if not os.path.exists(stat_path):
                                    os.makedirs(stat_path) 
                                matched_stat.to_csv(stat_path + 'overlapping.csv')
                                len_frame.to_csv(stat_path + 'length.csv')
                                
                                
                            except: 
                                print('Something wrong with stat: ' + match_mode + '/' + str(num_bins))




















