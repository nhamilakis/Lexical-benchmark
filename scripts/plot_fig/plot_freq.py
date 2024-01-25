#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot all figures in the paper

@author: jliu
"""
'''
color setting

human: orange
freq-based:purpl -> pink for matched
model:
    1.speech/prompted (the most human like): speech
    2.phone: blue
    2.phoneme/unprompted: purple

by_freq: similar colors but different shape
    high: line
    low: dotted
    
    
human is the most stressed
'''
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from plot_final_util import load_CDI,load_accum,load_exp,get_score_CHILDES,fit_curve






def plot_by_freq(vocab_type,human_dir,test_set_lst,accum_threshold,exp_threshold):
    
    '''
    merge all the curves into one figure
    '''
    
    sns.set_style('whitegrid')
    
    
    # get the ordered frequency list in descending order
    
    n = 0
    for file in os.listdir(human_dir):
        human_frame_all = pd.read_csv(human_dir + '/' + file)
        dataset = file[:-4]
        freq_lst = set(human_frame_all['group_original'].tolist())
        
    '''
        for freq in freq_lst:
            
            
            # read the results recursively
            
            human_frame = human_frame_all[human_frame_all['group_original']==freq]
            human_frame = human_frame.iloc[:, 5:-6]
            
            human_result = load_CDI(human_frame)
            ax = sns.lineplot(x="month", y="Proportion of acquired words", data=human_result, linewidth=3.5, label= freq)
            
        n+= 1
    '''
    
    n = 0
    while n < len(test_set_lst):    
        
    
        if vocab_type.split('[')[0] == 'recep':
            
            # accumulator model
            accum_all_freq = pd.read_csv(model_dir + test_set_lst[n] +'/accum.csv')
            freq_set = set(accum_all_freq['group'].tolist())
            freq_lst = [[string for string in freq_set if 'high' in string][0],[string for string in freq_set if 'low' in string][0]]
            for freq in freq_lst:   
                
                if freq.split('[')[0] == 'low':
                    style = 'dotted'
                else: 
                    style = 'solid'
            
                # plot accumulator model
               
                accum_all = accum_all_freq[accum_all_freq['group']==freq]
                accum_result = load_accum(accum_all,accum_threshold)
                ax = sns.lineplot(x="month", y="Lexical score", data=accum_result, color="Green", linestyle=style,linewidth=3, label = 'accum: ' + freq.split('[')[0])
        
                
            
                speech_result_all = pd.read_csv(model_dir + test_set_lst[n] + '/speech.csv')  
                # seelct speech model based on the freq band
                speech_result = speech_result_all[speech_result_all['group_median']==freq]
                ax = sns.lineplot(x="month", y="mean_score", data=speech_result, color='Blue',linewidth=3,linestyle=style, label= 'speech: ' + freq.split('[')[0])
                
          
            
            
                # plot phone-LSTM model
                phone_result_all = pd.read_csv(model_dir + test_set_lst[n] + '/phones.csv')
                # seelct speech model based on the freq band
                phone_result = phone_result_all[phone_result_all['group_median']==freq]
                ax = sns.lineplot(x="month", y="mean_score", data=phone_result,  color="Purple",linewidth=3,linestyle=style, label= 'phone: ' + freq.split('[')[0])
                
                
         
        elif vocab_type == 'exp':
            
            
            # unprompted generations
            for freq in freq_lst:   
                
                
                # CHILDES 
                target_dir = human_dir.replace('CDI', test_set)
                
                
                
                '''
                # add human-estimation here
                CHILDES_frame = pd.read_csv('Final_scores/Model_eval/' + lang + '/exp/CDI/CHILDES.csv')
                CHILDES_freq = CHILDES_frame[CHILDES_frame['group_original']==freq]
                CHIDES_result, avg_values = get_score_CHILDES(CHILDES_freq, exp_threshold)
                month_list_CHILDES = [int(x) for x in CHIDES_result.columns]
                ax = sns.lineplot(month_list_CHILDES, avg_values,
                                  linewidth=3, label=freq)
                
                
                
                '''
                # unprompted generation
                for file in os.listdir(target_dir):
                    target_frame = pd.read_csv(target_dir + '/' + file)
                    
                # read the generated file
                seq_frame_all = pd.read_csv(model_dir + test_set_lst[n] + '/unprompted.csv', index_col=0)
                # get the sub-dataframe by frequency  
                score_frame_all, avg_unprompted_all = load_exp(seq_frame_all,target_frame,True,exp_threshold)   
                
                word_group = target_frame[target_frame['group_original']==freq]['word'].tolist()
                score_frame = score_frame_all.loc[word_group]
                avg_values = score_frame.mean()
                month_list_unprompted = [int(x) for x in score_frame.columns]
                ax = sns.lineplot(month_list_unprompted, avg_values.values, linewidth=3, label= freq)
                
                
       
        n += 1
        
      
    # set the limits of the x-axis for each line
    for line in ax.lines:
        plt.xlim(0,36)
        plt.ylim(0,1)
    
    plt.title('{} {} vocab'.format(lang,vocab_type), fontsize=15, fontweight='bold')
    plt.xlabel('(Pseudo)age in month', fontsize=15)
    plt.ylabel('Proportion of children/models', fontsize=15)
    
    
    plt.tick_params(axis='both')
    
    # set legend location to avoid shade other curves
    if vocab_type == 'exp':
        legend_loc = 'upper left'
        
        # Create proxy artists for the legend labels
        
        plt.legend(fontsize='small',loc=legend_loc)
        
    else:
        legend_loc = 'upper right'
        plt.legend(fontsize='small', loc=legend_loc)
    
    plt.savefig('Final_scores/Figures/freq/' + lang + '_' + vocab_type + '.png',dpi=800)
    plt.show()
    

vocab_type = 'recep'    # recep or exp
lang = 'BE'    # AE or BE
test_set = 'CDI'   # CDI or Wuggy_filtered
exp_threshold = 60
accum_threshold = 100

vocab_type_lst = ['exp']
lang_lst = ['BE']
test_set_lst = ['matched']

for vocab_type in vocab_type_lst:
    for lang in lang_lst:
        for test_set in test_set_lst:

            # set paths
            model_dir = 'Final_scores/Model_eval/' + lang + \
                '/' + vocab_type + '/'
            human_dir = 'Final_scores/Human_eval/CDI/' + lang + '/' + vocab_type 

            plot_by_freq(vocab_type,human_dir,test_set_lst,accum_threshold,exp_threshold)



