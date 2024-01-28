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
import seaborn as sns
import matplotlib.pyplot as plt
import enchant
from plot_entropy_util import lemmatize,get_score

d = enchant.Dict("en_US")

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




def load_data(root_path, prompt_type,strategy,word_per_sec,hour,sec_frame):
    
    sns.set_style('whitegrid')
    
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
                    # append the list to the frame 
                    
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




# plot curves
def plot_average_curve(threshold,target_frame,CDI_frame,seq_frame_prompted,seq_frame_unprompted,target_type,title):
    
    sns.set_style('whitegrid')
    
    avg_values_unprompted_lst = []
    avg_values_prompted_lst = []
    # decompose the results by freq groups
    for freq in set(list(target_frame['group_original'].tolist())):
        word_group = target_frame[target_frame['group_original']==freq]
        # take the weighted average by the proportion of different frequency types
        score_frame_unprompted, avg_values_unprompted = get_score(word_group,seq_frame_unprompted,threshold)
        score_frame_prompted, avg_values_prompted = get_score(word_group,seq_frame_prompted,threshold)
        avg_values_unprompted_lst.append(avg_values_unprompted.values)
        avg_values_prompted_lst.append(avg_values_prompted.values)
        
    avg_unprompted = (avg_values_unprompted_lst[0] + avg_values_unprompted_lst[1]) / 2    
    avg_prompted = (avg_values_prompted_lst[0] + avg_values_prompted_lst[1]) / 2    
    
    # Plotting the line curve
    month_list_unprompted = [int(x) for x in score_frame_unprompted.columns]
    month_list_prompted = [int(x) for x in score_frame_prompted.columns]
    sns.lineplot(month_list_unprompted, avg_unprompted, label= 'Unprompted generation')
    sns.lineplot(month_list_prompted, avg_prompted, label= 'Prompted generation')
    
    # plot CDI data
    selected_words = CDI_frame.iloc[:, 5:-6]
        
    
    size_lst = []
    month_lst = []
    
    n = 0
    while n < selected_words.shape[0]:
        size_lst.append(selected_words.iloc[n])
        headers_list = selected_words.columns.tolist()
        month_lst.append(headers_list)
        n += 1

    size_lst_final = [item for sublist in size_lst for item in sublist]
    month_lst_final = [item for sublist in month_lst for item in sublist]
    month_lst_transformed = []
    for month in month_lst_final:
        month_lst_transformed.append(int(month))
    # convert into dataframe
    data_frame = pd.DataFrame([month_lst_transformed,size_lst_final]).T
    data_frame.rename(columns={0:'month',1:'Proportion of acquired words'}, inplace=True)
    data_frame_final = data_frame.dropna(axis=0)
    
    ax = sns.lineplot(x="month", y="Proportion of acquired words", data=data_frame_final, label= target_type)
    
    # set the limits of the x-axis for each line
    for line in ax.lines:
        plt.xlim(0,36)
        plt.ylim(0,1)
    
    plt.title('{} vocab (threshold = {})'.format(condition,threshold))
    plt.xlabel('month', fontsize=15)
    plt.ylabel('Proportion of children/models', fontsize=15)
    plt.tick_params(axis='both', labelsize=10)
    plt.legend()
    plt.savefig('Exp_figure/' + title + '_' + condition + '_' + str(threshold) + '.png',dpi=800)
    plt.show()




def plot_curve_freq(threshold,target_frame,CDI_frame,seq_frame, vocab_type,title,prompt_type):
    
    '''
    compare different frequency bands 
    '''
    
    
    sns.set_style('whitegrid')
    
    overlapping_words = [col for col in target_frame[vocab_type].tolist() if col in seq_frame.index.tolist()]
    # get the subdataframe
    selected_frame = seq_frame.loc[overlapping_words]
    
    # add more scores additional lines to the frame
    
    extra_word_lst = [element for element in target_frame[vocab_type].tolist() if element not in overlapping_words]
    
    for word in extra_word_lst:
        selected_frame.loc[word] = [0] * selected_frame.shape[1]
        
    # get the score based on theshold
    score_frame_all = selected_frame.applymap(lambda x: 1 if x >= threshold else 0)
    
    
    # same color for the same frequency bands; dotted line for human and dotted line for model
    color_lst = ['orange','blue']
    
    i = 0
    while i < len(set(target_frame['group_original'].tolist())):
        freq = list(set(target_frame['group_original'].tolist()))[i]
        word_group = target_frame[target_frame['group_original']==freq]['word'].tolist()
        score_frame = score_frame_all.loc[word_group]
            
        avg_values = score_frame.mean()
            
        # Plotting the line curve
        month_list = [int(x) for x in score_frame.columns]
        ax = sns.lineplot(month_list, avg_values.values, linestyle='dotted', color = color_lst[i],label='Model_' + freq)
    
        # plot CDI data
        
        all_words = CDI_frame[CDI_frame['group_original']==freq]
        
        selected_words = all_words.iloc[:, 5:-6]
        size_lst = []
        month_lst = []
        
        n = 0
        while n < selected_words.shape[0]:
            size_lst.append(selected_words.iloc[n])
            headers_list = selected_words.columns.tolist()
            month_lst.append(headers_list)
            n += 1
    
        size_lst_final = [item for sublist in size_lst for item in sublist]
        month_lst_final = [item for sublist in month_lst for item in sublist]
        month_lst_transformed = []
        for month in month_lst_final:
            month_lst_transformed.append(int(month))
        # convert into dataframe
        data_frame = pd.DataFrame([month_lst_transformed,size_lst_final]).T
        data_frame.rename(columns={0:'month',1:'Proportion of acquired words'}, inplace=True)
        data_frame_final = data_frame.dropna(axis=0)
        
        ax = sns.lineplot(x="month", y="Proportion of acquired words", data=data_frame_final, color = color_lst[i], linestyle='solid', label=title + '_'+ freq)
    
    
        i += 1
    # set the limits of the x-axis for each line
    for line in ax.lines:
        plt.xlim(0,36)
        plt.ylim(0,1)
    
    plt.title('{} vocab ({} generation,threshold = {})'.format(condition,prompt_type,threshold))
    plt.xlabel('month', fontsize=15)
    plt.ylabel('Proportion of children/models', fontsize=15)
    plt.tick_params(axis='both', labelsize=10)
    plt.legend()
    plt.savefig('Exp_figure/' + title + '_' + prompt_type + '_' + str(threshold) + '.png',dpi=800)
    plt.show()


root_path = 'generation'
strategy = 'sample_random'
word_per_sec = 3
hour = 10

by_freq = True
vocab_type = 'word'   

condition = 'Expressive'


sec_frame = pd.read_csv('/data/Lexical-benchmark/vocal_month.csv')

# load data
seq_frame_prompted,lemma_frame_prompted = load_data(root_path, 'prompted',strategy,word_per_sec,hour,sec_frame)
seq_frame_unprompted,lemma_frame_unprompted = load_data(root_path, 'unprompted',strategy,word_per_sec,hour,sec_frame)


save_root = '/data/Lexical-benchmark/exp/reference/generation_stat/'
seq_frame_prompted.to_csv(save_root + 'prompted.csv')     
seq_frame_unprompted.to_csv(save_root + 'unprompted.csv')  

seq_frame_prompted1= pd.read_csv('prompted.csv', index_col=0)
seq_frame_prompted1 = seq_frame_prompted1.rename_axis('Index')


seq_frame_unprompted2= pd.read_csv('unprompted.csv')
seq_frame_unprompted2 = seq_frame_unprompted2.rename_axis('Index')
word_path = 'target_words'
AE_CDI = pd.read_csv(word_path + '/AE_exp.csv')
BE_CDI = pd.read_csv(word_path + '/BE_exp.csv')
matched_AE_CDI = pd.read_csv(word_path + '/matched_AE_exp.csv')
matched_BE_CDI = pd.read_csv(word_path + '/matched_BE_exp.csv')

threshold = 1
CDI_frame = AE_CDI
target_frame = AE_CDI 
target_type = 'AE_CDI'
title = 'AE_CDI'  

plot_average_curve(threshold,target_frame,CDI_frame,seq_frame_prompted1,seq_frame_unprompted2,target_type,title)


plot_curve_freq(threshold,target_frame,CDI_frame,seq_frame_prompted, vocab_type,title,'prompted')

plot_curve_freq(threshold,target_frame,CDI_frame,seq_frame_unprompted, vocab_type,title,'unprompted')

