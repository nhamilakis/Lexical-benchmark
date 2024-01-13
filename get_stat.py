#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get basic stat of different sets

different datasets: 
    - audiobook: intersection between wuggy sets and words that appear at least once in each chunk
    - AO-CHILDES: existing scripts
    - CELEX: SUBLEX-US     
@author: jliu
two versions of freq: char and phoneme

"""
import sys
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
from stat_util import get_freq_table, convert_to_log, match_range, get_equal_bins, match_bin_range
from aochildes.dataset import AOChildesDataSet


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Investigate CHILDES corpus')
    
    parser.add_argument('--lang', type=str, default = 'AE',
                        help='langauges to test: AE, BE or FR')
    
    parser.add_argument('--eval_condition', type=str, default = 'recep',
                        help='which type of words to evaluate; recep or exp')
    
    parser.add_argument('--CDIPath', type=str, default = 'Final_scores/Human_eval/CDI/',
                        help='path to the evaluation material; one of the variables to invetigate')
    
    parser.add_argument('--freqPath', type=str, default = 'stat/',
                        help='path to the evaluation material; one of the variables to invetigate')
    
    parser.add_argument('--word_format', type=str, default = 'char',
                        help='char or phon format')
    
    parser.add_argument('--num_bins', type=int, default = 2,
                        help='number of eaqul-sized bins')
    
    parser.add_argument('--freq_type', type=str, default = 'freq',
                        help='different freq types: freq or log_freq')
    
    parser.add_argument('--match_mode', type=str, default = 'range_aligned',
                        help='different freq types: range_aligned, bin_range_aligned or density_aligned')
    
    return parser.parse_args(argv)

   
     

def get_freq_frame(test,train_path):
    
    '''
    get the overall freq of CHILDES and train data fre
    ''' 
    
    # check whether the overall freq file exists
    if os.path.exists(train_path + 'Audiobook_fre_table.csv'):
        audiobook_fre_table = pd.read_csv(train_path + 'Audiobook_fre_table.csv') 
        
    else:
        # calculate fre table respectively
        with open(train_path + 'Audiobook_train.txt', encoding="utf8") as f:
            Audiobook_lines = f.readlines()
            
        audiobook_fre_table = get_freq_table(Audiobook_lines) 
        audiobook_fre_table.to_csv(train_path + 'Audiobook_fre_table.csv')
      
        
    if os.path.exists(train_path + 'CHILDES_fre_table.csv'):
        CHILDES_fre_table = pd.read_csv(train_path + 'CHILDES_fre_table.csv') 
        
    else:
        # calculate fre table respectively
        CHILDES_lines = AOChildesDataSet().load_transcripts() 
        CHILDES_fre_table = get_freq_table(CHILDES_lines) 
        CHILDES_fre_table.to_csv(train_path + 'CHILDES_fre_table.csv')    
        
    
    CELEX = pd.read_excel(train_path + 'SUBTLEX_US.xls')
    
    # Find the intersection of the three lists
    selected_words = list(set(test['word'].tolist()).intersection(set(CHILDES_fre_table['Word'].tolist()),set(audiobook_fre_table['Word'].tolist()), set(CELEX['Word'].tolist())))
    
    # get the dataframe with all the dataset
    freq_frame = pd.DataFrame([selected_words]).T
    freq_frame.rename(columns={0:'word'}, inplace=True)
    
    audiobook_lst = []
    CELEX_lst = []
    CHILDES_lst = []
    
    for word in selected_words:
        
        audiobook_lst.append(audiobook_fre_table[audiobook_fre_table['Word']==word]['Norm_freq_per_million'].item())
        CELEX_lst.append(CELEX[CELEX['Word']==word]['SUBTLWF'].item()) 
        CHILDES_lst.append(CHILDES_fre_table[CHILDES_fre_table['Word']==word]['Norm_freq_per_million'].item()) 
    
    freq_frame['Audiobook_freq_per_million'] = audiobook_lst
    freq_frame['CELEX_freq_per_million'] = CELEX_lst 
    freq_frame['CHILDES_freq_per_million'] = CHILDES_lst
    
    # get log 10
    freq_frame['CELEX_log_freq_per_million'] = freq_frame['CELEX_freq_per_million'].apply(convert_to_log)
    freq_frame['Audiobook_log_freq_per_million'] = freq_frame['Audiobook_freq_per_million'].apply(convert_to_log)
    freq_frame['CHILDES_log_freq_per_million'] = freq_frame['CHILDES_freq_per_million'].apply(convert_to_log)
    
    return freq_frame




def match_freq(CDI,audiobook,match_mode,num_bins):
    
    '''
    match machine CDI with human CDI based on different match modes
    return {condi:[matched dataframes,bins,bin_stat]} 
    '''
    freq_dict = {}
    
    matched_CDI,matched_audiobook = match_range(CDI,audiobook)
    CDI_bins, CDI_bins_stats = get_equal_bins(CDI,num_bins)
    
    if match_mode == 'range_aligned':
        audiobook_bins, audiobook_bins_stats,matched_audiobook = get_equal_bins(audiobook,num_bins)
    
    elif match_mode == 'bin_range_aligned':
        audiobook_bins, audiobook_bins_stats,matched_audiobook = match_bin_range(CDI_bins,audiobook)
    
    
    # compare the results
    compare_histogram(matched_CDI,matched_audiobook,n,freq_type,lang,eval_condition,word_format,freqPath,match_mode)
    
    
    
    return bin_stats
    




def compare_histogram(matched_CDI,matched_audiobook,n,freq_type,lang,eval_condition,word_format,freqPath,match_mode):  
    
    # remove rows of low-freqeuncy words 
    
    audiobook = matched_audiobook['Audiobook_'+ freq_type + '_per_million'].tolist()
    
    # select corresponding bins
    
    CHILDES = matched_CDI['CHILDES_'+ freq_type + '_per_million'].tolist()
    
    target_bins, bins_stats_CDI = plot_density_hist(CHILDES,n,label = 'CDI') 
    
    
    plt.xlabel(freq_type + '_per million')
    plt.ylabel('Density')
    
    # set the limits of the x-axis for each line
    if freq_type == 'freq':
        
        plt.xlim(0,850)
        plt.ylim(0,0.035)
            
    elif freq_type == 'log_freq':
        
        plt.xlim(-1,4)
        plt.ylim(0,1.5)
    
    plt.title(lang + ' ' + eval_condition + ' (' + freq_type + ', '  + word_format + ')')
    plt.legend() 
    
    # save the plot to the target dir
    OutputPath = freqPath + 'fig/' + str(n) + '/' + match_mode + '/'
    if not os.path.exists(OutputPath):
        os.makedirs(OutputPath) 
        
    plt.savefig(OutputPath + 'Compare_' + lang + '_' + eval_condition + '_' + freq_type, dpi=800)
    plt.clf()
    
    
    
    
    return bins_stats




def plot_all_histogram(freq,n,freq_type,lang,eval_condition, eval_type, word_format,freqPath):  
    
    '''
    plot both the tested 
    
    n is num_bins in eaqul-sized 
    '''
    
    audiobook = freq['Audiobook_'+ freq_type + '_per_million'].tolist()
    CELEX = freq['CELEX_'+ freq_type + '_per_million'].tolist()
    CHILDES = freq['CHILDES_'+ freq_type + '_per_million'].tolist()
    # plot freq per million  
    plot_density_hist(CHILDES,n,label = 'CHILDES') 
    plot_density_hist(audiobook,n,label = 'Audiobook')
    plot_density_hist(CELEX,n,label = 'CELEX')     
    
    plt.xlabel(freq_type + '_per million')
    plt.ylabel('Density')
    
    # set the limits of the x-axis for each line
    if freq_type == 'freq':
        
        plt.xlim(0,850)
        plt.ylim(0,0.035)
            
    elif freq_type == 'log_freq':
        
        plt.xlim(-1,4)
        plt.ylim(0,1.5)
    
    plt.title(lang + ' ' + eval_condition + ' ' +  eval_type + ' (' + freq_type + ', '  + word_format + ')')
    plt.legend() 
    
    # save the plot to the target dir
    OutputPath = freqPath + 'fig/' + str(n) + '/'
    if not os.path.exists(OutputPath):
        os.makedirs(OutputPath) 
        
    plt.savefig(OutputPath + eval_type + '_' + lang + '_' + eval_condition + '_' + freq_type, dpi=800)
    plt.clf()
    
n = 4
freq_type = 'log_freq' 
match_mode = 'aligned'
lang = 'AE'
eval_condition = 'exp'
word_format = 'char'


def main(argv):
    
    # Args parser
    args = parseArgs(argv)
    
    if not os.path.exists(args.freqPath + 'freq/'+ args.word_format + '/'):
        os.makedirs(args.freqPath + 'freq/'+ args.word_format + '/') 
    
    
    # step 1: load overall freq data
    # skip counting words if there already exists the file
    CDI_freqOutput = args.freqPath + 'freq/'+ args.word_format + '/CDI_' + args.lang + '_' + args.eval_condition + '.csv'
    matched_freqOutput = args.freqPath + 'freq/'+ args.word_format + '/matched_' + args.lang + '_' + args.eval_condition + '.csv'
    train_path = args.freqPath + 'corpus/'+ args.word_format + '/'
    
    
    # check whether there exists the matched data
    
    if os.path.exists(CDI_freqOutput):
        print('There exists the CDI freq words')
        CDI = pd.read_csv(CDI_freqOutput) 
    
    else:
        testPath = args.CDIPath + args.lang + '/' + args.eval_condition
        for file in os.listdir(testPath):
            CDI_test = pd.read_csv(testPath + '/' + file)
        
        CDI = get_freq_frame(CDI_test,train_path)
        
      
    if os.path.exists(matched_freqOutput):
        print('There exists the audiobook freq words')
        audiobook = pd.read_csv(matched_freqOutput) 
    
    else:
        audiobook_test = pd.read_csv(train_path + '/Audiobook_test.csv')
        audiobook = get_freq_frame(audiobook_test,train_path)
        
        
        
    # step 2: match the freq
    matched_CDI,matched_audiobook = match_freq(CDI,audiobook)
    # save the filtered results
    matched_CDI.to_csv('stat/freq/char/CDI_' + args.lang + '_' + args.eval_condition  +'.csv')
    matched_audiobook.to_csv('stat/freq/char/matched_' + args.lang + '_' + args.eval_condition  +'.csv')    
    
    
    
    # step 3: plot the distr figures
    
    # plot out the matched freq results
    bins_stats = compare_histogram(matched_CDI,matched_audiobook,args.num_bins,args.freq_type,args.lang,args.eval_condition,args.word_format,args.freqPath,args.match_mode)
    # save the freq stat
    matched_path = 'stat/freq/char/stat/compare.csv'
    if os.path.exists(matched_path):
        bins_frame = pd.read_csv(matched_path)
        bins_frame = pd.concat([bins_stats,bins_frame])
        # only preserve the newest version
        bins_frame = bins_frame.drop_duplicates()
    else:
        bins_frame = bins_stats
        
    bins_frame.to_csv(matched_path)    

    
    '''
    plot_all_histogram(matched_CDI,args.num_bins,args.freq_type,args.lang,args.eval_condition,'CDI',args.word_format,args.freqPath)
    plot_all_histogram(matched_audiobook,args.num_bins,args.freq_type,args.lang,args.eval_condition,'matched',args.word_format,args.freqPath)
    '''

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)