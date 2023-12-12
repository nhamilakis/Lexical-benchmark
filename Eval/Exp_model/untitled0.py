#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count the numbwer of generated words from the model
@author: jliu
"""

import os
import pandas as pd
from plot_entropy_util import plot_single_para, plot_distance, match_seq,lemmatize,filter_words
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


def load_data(root_path):
    
    '''
    count all the words in the generated tokens 
    
    input: the root directory containing all the generated files adn train reference
    output: 1.the info frame with all the generarted tokens
            2.the reference frame with an additional column of the month info
            3.vocab size frame with the seq word and lemma frequencies
    '''
    
    # load the rerference data
    frame_all = []
    seq_all = []
    month_lst = []
    prompt_lst = []
    h_all = []
    prob_all =[]
    strategy_lst = []
    beam_lst = []
    topk_lst = []
    topp_lst = []
    random_lst = []
    temp_lst = []
    directory_lst = []
    chunk_lst = []
    # go over the generated files recursively
    
    for month in os.listdir(root_path): 
        
        for chunk in os.listdir(root_path + '/' + month): 
            
            for prompt_type in os.listdir(root_path + '/' + month + '/' + chunk): 
                
                for strategy in os.listdir(root_path + '/' + month + '/' + chunk + '/' + prompt_type): 
                        
                    for file in os.listdir(root_path + '/' + month + '/' +  chunk+ '/' + prompt_type + '/' + strategy):
                            
                        # load decoding strategy information       
                        data = pd.read_csv(root_path + '/' + month + '/' +  chunk+ '/' + prompt_type + '/' + strategy + '/' + file)
                                    
                        try:
                                        
                         # count words
                             seq = []
                             n = 0
                             while n < data.shape[0]:
                                 generated = data['LSTM_segmented'].tolist()[n].split(' ')
                                 seq.extend(generated)
                                 n += 1
                                        
                             # get freq lists
                             frequencyDict = collections.Counter(seq)  
                             freq_lst = list(frequencyDict.values())
                             word_lst = list(frequencyDict.keys())
                             fre_table = pd.DataFrame([word_lst,freq_lst]).T
                                        
                             col_Names=["Word", "Freq"]
                                        
                             fre_table.columns = col_Names
                             seq_all.extend(seq)
                                        
                                        
                             if strategy == 'beam':
                                 beam_lst.append(file.split('_')[0])
                                 topk_lst.append('0')
                                 topp_lst.append('0')
                                 random_lst.append('0')
                                 strategy_lst.append(strategy)
                                 fre_table['BEAM'] = file.split('_')[0]
                                 fre_table['TOPK'] ='0'
                                 fre_table['TOPP'] ='0'
                                 fre_table['RANDOM'] ='0'
                                            
                                            
                             elif strategy == 'sample_topk':
                                 topk_lst.append(file.split('_')[0])
                                 beam_lst.append('0')
                                 topp_lst.append('0')
                                 random_lst.append('0')
                                 strategy_lst.append(strategy.split('_')[1])
                                 fre_table['TOPK'] = file.split('_')[0]
                                 fre_table['BEAM'] ='0'
                                 fre_table['TOPP'] ='0'
                                 fre_table['RANDOM'] ='0'
                                            
                                            
                                            
                             elif strategy == 'sample_topp':
                                 topp_lst.append(file.split('_')[0])
                                 beam_lst.append('0')
                                 topk_lst.append('0')
                                 random_lst.append('0')
                                 strategy_lst.append(strategy.split('_')[1])
                                 fre_table['TOPP'] = file.split('_')[0]
                                 fre_table['BEAM'] ='0'
                                 fre_table['TOPK'] ='0'
                                 fre_table['RANDOM'] ='0'
                                            
                                            
                             elif strategy == 'sample_random':
                                 random_lst.append('1')
                                 topk_lst.append('0')
                                 beam_lst.append('0')
                                 topp_lst.append('0')
                                 strategy_lst.append(strategy.split('_')[1])
                                 fre_table['RANDOM'] = file.split('_')[0]
                                 fre_table['BEAM'] ='0'
                                 fre_table['TOPP'] ='0'
                                 fre_table['TOPK'] ='0'
                                        
                            # concatnete all the basic info regarding the genrated seq
                             fre_table['MONTH'] = month
                             fre_table['PROMPT'] = prompt_type
                             fre_table['TEMP'] = float(file.split('_')[1])
                             fre_table['CHUNK'] = chunk
                             fre_table['DECODING'] = strategy.split('_')[-1]
                             prompt_lst.append(prompt_type)
                             month_lst.append(month)
                             temp_lst.append(float(file.split('_')[1]))
                             directory_lst.append(month+ '/' + prompt_type + '/' + strategy + '/' + file)
                             chunk_lst.append(chunk)
                             frame_all.append(fre_table)
                             
                             print('SUCCESS: ' + month+ '/' + prompt_type + '/' + strategy + '/' + file)    
                                    
                        except:
                             print('FAILURE: ' + month+ '/' + prompt_type + '/' + strategy + '/' + file)
                                
                    
    info_frame = pd.DataFrame([month_lst,chunk_lst,prompt_lst,strategy_lst,beam_lst,topk_lst,topp_lst,random_lst,temp_lst,directory_lst]).T
    
    # rename the columns
    info_frame.rename(columns = {0:'month', 1:'chunk', 2:'prompt',3:'decoding', 4:'beam', 5:'topk', 6:'topp',7:'random', 8:'temp',9:'entropy',10:'prob',11:'location'}, inplace = True)
   
    # remove the row with NA values
    info_frame = info_frame.dropna()
    info_frame = info_frame[(info_frame['random'] != '.') & (info_frame['topp'] != '.') & (info_frame['topk'] != '.')]
    
    
    # sort the result based on temperature to get more organized legend labels
    info_frame = info_frame.sort_values(by='temp', ascending=True)
    
    # # get word count and lemma count frames
    seq_lst = list(set(seq_all))
    
    seq_frame = match_seq(seq_lst,frame_all)
    
    word_lst, lemma_dict = lemmatize(seq_lst)
    
    word_lst.extend(['MONTH','CHUNK','PROMPT','BEAM','TOPK','TEMP','TOPP','RANDOM'])
    word_frame = seq_frame[word_lst]
    
    # reshape the lemma frame based onthe word_frame: basic info, lemma, total counts
    lemma_frame = seq_frame[['MONTH','CHUNK','PROMPT','BEAM','TOPK','TEMP','TOPP','RANDOM']]
    for lemma, words in lemma_dict.items():
        # Merge columns in the list by adding their values
        lemma_frame[lemma] = word_frame[words].sum(axis=1)
    
    return info_frame, seq_frame, word_frame, lemma_frame
   


# check the overlap between wordbank and those for receptive vocab

vocab_filtered = pd.read_csv(root_path + '/vocab_filtered.csv')
vocab_unfiltered = pd.read_csv(root_path + '/vocab_unfiltered.csv')
AE_CDI = pd.read_csv(root_path + '/AE_content.csv')
BE_CDI = pd.read_csv(root_path + '/BE_content.csv')

threshold = 1
target_words = BE_CDI['words'].tolist()
target_words = word_frame.columns.tolist()
# plot the vocab size curve based on word counts
def plot_curve(threshold,word_frame, target_words):
    
    
    '''
    input: the frame with word vocab size score

    Returns
    the learning curve of the figure

    '''
    def filter_words(word_frame, target_words):
    
        '''
        input: 1.word count dataframe 
               2.a list of the selected words from expressive vocabulary
    
        Returns
                1. target word count dataframe across different months
                2. dataframe with vocab/word size 
        '''
    
        overlapping_words = [col for col in target_words if col in word_frame.columns]
        # append other info in the selected vocab
        overlapping_words.extend(['MONTH','CHUNK','PROMPT','BEAM','TOPK','TEMP','TOPP','RANDOM'])
        # Selecting existing columns based on the list of column names
        
        selected_frame = word_frame[overlapping_words]
    
        return selected_frame

    selected_frame = filter_words(word_frame, target_words)
    
    # remove duplicated 
    word_size = get_score(threshold,selected_frame)
    
    # remove duplicated columns 
    word_size = word_size.loc[:, ~word_size.columns.duplicated()]

    target_frame = word_size[word_size['TOPP'] != '0']   

    month_lst = []              
    for month in target_frame['MONTH'].tolist():
        pseudo_month = int(month[:-1])/89
        month_lst.append(pseudo_month)
        
    target_frame['Pseudo_month'] = month_lst
    
    target_frame['Proportion of model'] = target_frame['vocab_size']/selected_frame.shape[1]
    
    sns.set_style('whitegrid')
    ax = sns.lineplot(data=target_frame, x='Pseudo_month', y='Proportion of model')
                    
    plt.title('Unprompted generation', fontsize=20)
    plt.show()



plot_curve(threshold,word_frame, BE_CDI['words'].tolist())


  
# plot the results by different frequency bands

def plot_by_freq(threshold,word_frame, target_words):



# investigate POS of the generated words/ The emergence of function words 






    
    
def main(argv):
    
    # Args parser
    args = parseArgs(argv)
    
    TextPath = args.TextPath
    condition = args.condition
    lang = args.lang
    OutputPath = args.OutputPath + '/' + lang + '/' + condition
    eval_path = args.eval_path + lang + '/' + condition
    hour = args.hour
    threshold_range = args.threshold_range
    word_per_hour = args.word_per_hour
    rmse_frame_all = pd.DataFrame()
    
    # step 1: load data and count words
    month_stat = load_data(TextPath,OutputPath,lang,condition)
    freq_frame = count_words(OutputPath,month_stat,eval_path,hour,word_per_hour)
    
    # step 2: get the score based on different thresholds
    for threshold in threshold_range:
    
        score_frame = get_score(freq_frame,OutputPath,threshold,hour)
             
        # plot the developmental bars and calculate the fitness
        print('Plotting the developmental bars')
             
        # compare and get the best combination of threshold two variables
        rmse = plot_curve(OutputPath,eval_path,score_frame,threshold,month_stat,condition)
        rmse_frame_temp = pd.DataFrame([threshold, rmse]).T
        rmse_frame = rmse_frame_temp.rename(columns={0: "Chunksize", 1: "threshold", 2: "rmse" })    
        rmse_frame_all = pd.concat([rmse_frame_all,rmse_frame])
        
       
    rmse_frame_all.to_csv(OutputPath + '/Scores/Fitness_All.csv')  
    
    

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    
    