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
from plot_final_util import load_CDI,load_accum,load_exp,get_score_CHILDES
import numpy as np


# Using a predefined color palette with a blue-green shade


def plot_all(vocab_type, human_dir, test_set, accum_threshold, exp_threshold):

    sns.set_style('whitegrid')

    # plot the curves respectively
    # plot human
    linestyle_lst = ['solid', 'dotted']
    n = 0
    for file in os.listdir(human_dir):
        human_frame = pd.read_csv(human_dir + '/' + file).iloc[:, 5:-6]
        dataset = file[:-4]
        human_result = load_CDI(human_frame)
        ax = sns.lineplot(x="month", y="Proportion of acquired words", data=human_result,
                          color="Red", linestyle=linestyle_lst[n], linewidth=3.5, label='CDI_' + dataset)
        n += 1

    # if receptive vocab, load speech-based and phones-based models
    # plot speech-based model
    if vocab_type == 'recep':

        # plot accumulator model
        accum_all = pd.read_csv(model_dir + 'accum.csv')
        accum_result = load_accum(accum_all, accum_threshold)
        ax = sns.lineplot(x="month", y="Lexical score", data=accum_result,
                          color="Green", linewidth=3, label='Accumulator')

        speech_result = pd.read_csv(model_dir + 'speech.csv')
        ax = sns.lineplot(x="month", y="mean_score", data=speech_result,
                          color='Blue', linewidth=3, label='LSTM-speech')

        # plot speech-based model
        phones_result = pd.read_csv(model_dir + 'phones.csv')
        ax = sns.lineplot(x="month", y="mean_score", data=phones_result,
                          color="Purple", linewidth=3, label='LSTM-phones')

    elif vocab_type == 'exp':

        # group by different frequencies

        target_dir = human_dir.replace('CDI', test_set)

        # add human-estimation here
        CHILDES_freq = pd.read_csv(model_dir + '/CHILDES.csv')
        CHIDES_result, avg_values = get_score_CHILDES(CHILDES_freq, 200)
        month_list_CHILDES = [int(x) for x in CHIDES_result.columns]
        ax = sns.lineplot(month_list_CHILDES, avg_values,
                          color="Orange", linewidth=3, label='CHILDES-estimation')

        # unprompted generation
        for file in os.listdir(target_dir):
            target_frame = pd.read_csv(target_dir + '/' + file)

        seq_frame_all = pd.read_csv(model_dir + 'unprompted.csv', index_col=0)
        score_frame_unprompted, avg_unprompted = load_exp(
            seq_frame_all, target_frame, False, exp_threshold)
        month_list_unprompted = [int(x)
                                 for x in score_frame_unprompted.columns]
        ax = sns.lineplot(month_list_unprompted, avg_unprompted,
                          color="Grey", linewidth=3, label='LSTM-unprompted')

        '''
        # prompted generation
        seq_frame_all = pd.read_csv(model_dir + 'prompted.csv', index_col=0)   
        score_frame_prompted, avg_prompted = load_exp(seq_frame_all,target_frame,False,exp_threshold)
        month_list_prompted = [int(x) for x in score_frame_prompted.columns]
        ax = sns.lineplot(month_list_prompted, avg_prompted,  color="Blue",linewidth=3, label='LSTM-prompted')
        '''

    # set the limits of the x-axis for each line
    for line in ax.lines:
        plt.xlim(0, 36)
        plt.ylim(0, 1)

    plt.title('{} {} vocab ({} set)'.format(
        lang, vocab_type, test_set), fontsize=15, fontweight='bold')
    plt.xlabel('(Pseudo)age in month', fontsize=15)
    plt.ylabel('Proportion of children/models', fontsize=15)

    plt.tick_params(axis='both', labelsize=10)

    if vocab_type == 'recep':
        legend_loc = 'upper right'

    elif vocab_type == 'exp':
        legend_loc = 'upper left'

    plt.legend(loc=legend_loc)

    plt.savefig('Final_scores/Figures/avg/' + lang + '_' +
                vocab_type + '_' + test_set+'.png', dpi=800)
    plt.show()


# receptive vocab
vocab_type = 'recep'    # recep or exp
lang = 'AE'    # AE or BE
test_set = 'CDI'   # CDI or Wuggy_filtered
exp_threshold = 200

vocab_type_lst = ['recep', 'exp']
lang_lst = ['AE', 'BE']
test_set_lst = ['CDI']

vocab_type_lst = ['recep']
lang_lst = ['AE']
test_set_lst = ['CDI']

for vocab_type in vocab_type_lst:
    for lang in lang_lst:
        for test_set in test_set_lst:

            # set paths
            model_dir = 'Final_scores/Model_eval/' + lang + \
                '/' + vocab_type + '/' + test_set + '/'
            human_dir = 'Final_scores/Human_eval/CDI/' + lang + '/' + vocab_type

            if vocab_type == 'recep':
                accum_threshold = 200
            elif vocab_type == 'exp':
                accum_threshold = 500

            plot_all(vocab_type, human_dir, test_set,
                     accum_threshold, exp_threshold)




def get_result(vocab_type,human_dir,test_set_lst,accum_threshold,exp_threshold):
    
    '''
    get the mean scores of different conditions
    
    input: different settings
    output: the mean score frame with different settings
    '''
    # get average scores of different test sets: mean and range
    
    score_frame = pd.DataFrame()
    
    
    for vocab_type in vocab_type_lst:
        if vocab_type ==  'recep':
            accum_threshold = 200
        elif vocab_type ==  'exp':
            accum_threshold = 500
        
        for lang in lang_lst:
            for test_set in test_set_lst:
                
                # set paths
                model_dir = 'Final_scores/Model_eval/' + lang + '/' + vocab_type + '/' + test_set + '/'
                
                accum_all = pd.read_csv(model_dir + 'accum.csv')
                # select the corresponding threshold
                accum_result = accum_all[accum_all['threshold']==accum_threshold]
                accum_score  = accum_result.groupby('group')['Lexical score'].agg(['mean', 'min', 'max']).reset_index()
                # rename the column header by index
                accum_score = accum_score.rename(columns={'group': 'group_median'})+ '_' + test_model+ 
                accum_score['model'] = 'accum'
                
                if vocab_type == 'recep':
                    # read different files to get mean score
                    speech_result = pd.read_csv(model_dir + 'speech.csv')
                    phones_result = pd.read_csv(model_dir + 'phones.csv')
                
                    # Grouping by 'Category' and calculating mean, max, and min of 'Values'
                    speech_score  = speech_result.groupby('group_median')['mean_score'].agg(['mean', 'min', 'max']).reset_index()
                    phones_score  = phones_result.groupby('group_median')['mean_score'].agg(['mean', 'min', 'max']).reset_index()
                
                    speech_score['model'] = 'speech'
                    phones_score['model'] = 'phones'
                    # concatenate the scores together
                    result_frame = pd.concat([accum_score, speech_score, phones_score], axis=0)
                    result_frame['lang'] = lang
                    result_frame['test_set'] = test_set
                    result_frame['vocab_type'] = vocab_type
                    
                elif vocab_type == 'exp':
                    
                    
                    human_dir = 'Final_scores/Human_eval/CDI/' + lang + '/' + vocab_type 
                    for file in os.listdir(human_dir):
                        human_frame_all = pd.read_csv(human_dir + '/' + file)
                    
                    # get the ordered frequency list in descending order
                    freq_set = set(human_frame_all['group_original'])
                    freq_lst = [[string for string in freq_set if 'high' in string][0],[string for string in freq_set if 'low' in string][0]]
                    
                    result_frame = pd.DataFrame()
                    for freq in freq_lst:   
                        
                        ref_path = human_dir
                    
                        target_dir = ref_path.replace('CDI', test_set)
                        
                        # unprompted generation
                        for file in os.listdir(target_dir):
                            target_frame = pd.read_csv(target_dir + '/' + file)
                            
                        # read the generated file
                        seq_frame_all = pd.read_csv(model_dir + 'unprompted.csv', index_col=0)
                        # get the sub-dataframe by frequency  
                        score_frame_all, avg_unprompted_all = load_exp(seq_frame_all,target_frame,True,exp_threshold)   
                        
                        word_group = target_frame[target_frame['group_original']==freq]['word'].tolist()
                        score_frame_unprompted = score_frame_all.loc[word_group]
                        avg_unprompted = score_frame_unprompted.mean()
                        # avg among differetn months
                        unprompted_score = pd.DataFrame([np.mean(avg_unprompted),np.min(avg_unprompted),np.max(avg_unprompted)]).T
                        # rename the frame
                        unprompted_score.columns = ['mean', 'min', 'max']
                        unprompted_score['model'] = 'unprompted'
                        
                        # read the generated file
                        seq_frame_all = pd.read_csv(model_dir + 'prompted.csv', index_col=0)
                        # get the sub-dataframe by frequency  
                        score_frame_all, avg_unprompted_all = load_exp(seq_frame_all,target_frame,True,exp_threshold)   
                        
                        word_group = target_frame[target_frame['group_original']==freq]['word'].tolist()
                        score_frame_unprompted = score_frame_all.loc[word_group]
                        avg_unprompted = score_frame_unprompted.mean()
                        # avg among differetn months
                        prompted_score = pd.DataFrame([np.mean(avg_unprompted),np.min(avg_unprompted),np.max(avg_unprompted)]).T
                        # rename the frame
                        prompted_score.columns = ['mean', 'min', 'max']
                        prompted_score['model'] = 'prompted'
                        
                        result_frame_temp = pd.concat([accum_score,unprompted_score, prompted_score], axis=0)
                        result_frame_temp['lang'] = lang
                        result_frame_temp['test_set'] = test_set
                        result_frame_temp['vocab_type'] = vocab_type
                        result_frame_temp['group_median'] = freq
                        # concatenate the result together
                        result_frame = pd.concat([result_frame,result_frame_temp])
                        
                score_frame = pd.concat([score_frame,result_frame])
                
                
    return score_frame        



vocab_type =  'recep'    # recep or exp
lang = 'AE'    # AE or BE
test_set = 'CDI'   # CDI or Wuggy_filtered
exp_threshold = 200

vocab_type_lst = ['recep','exp'] 

lang_lst = ['AE','BE']
test_set_lst = ['CDI','matched']

score_frame = get_score(vocab_type,human_dir,test_set_lst,accum_threshold,exp_threshold)

# group in different freq bands
# Grouping by multiple columns and calculating mean scores
grouped_data = score_frame.groupby(['model','lang','test_set','vocab_type' ])[['mean', 'min','max']].mean().reset_index()
grouped_data.to_csv('result.csv')




def plot_by_freq(vocab_type,human_dir,test_set_lst,accum_threshold,exp_threshold):
    
    '''
    merge all the curves into one figure
    '''
    
    sns.set_style('whitegrid')
    
    color_lst = ['Red','Pink']
    
    # get the ordered frequency list in descending order
    
    
    n = 0
    for file in os.listdir(human_dir):
        human_frame_all = pd.read_csv(human_dir + '/' + file)
        dataset = file[:-4]
        freq_set = set(human_frame_all['group_original'].tolist())
        freq_lst = [[string for string in freq_set if 'high' in string][0],[string for string in freq_set if 'low' in string][0]]
        for freq in freq_lst:
            
            if freq.split('[')[0] == 'low':
                style = 'dotted'
            else: 
                style = 'solid'
            
            # read the results recursively
            
            human_frame = human_frame_all[human_frame_all['group_original']==freq]
            human_frame = human_frame.iloc[:, 5:-6]
            
            human_result = load_CDI(human_frame)
            ax = sns.lineplot(x="month", y="Proportion of acquired words", data=human_result, color=color_lst[n], linestyle=style,linewidth=3.5, label= dataset + ': ' + freq.split('[')[0])
            
        n+= 1
    
    
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
            
            # CHILDES estimation data
            
            
            
            # unprompted generations
            for freq in freq_lst:   
                
                ref_path = human_dir
                
                if freq.split('[')[0] == 'low':
                    style = 'dotted'
                else: 
                    style = 'solid'
            
                target_dir = ref_path.replace('CDI', test_set)
                    
                
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
                ax = sns.lineplot(month_list_unprompted, avg_values.values, color="Grey",linestyle=style, linewidth=3, label='character: ' + freq.split('[')[0])
           
        
       
        n += 1
        
        
    # set the limits of the x-axis for each line
    for line in ax.lines:
        plt.xlim(0,36)
        plt.ylim(0,1)
    
    plt.title('{} {} vocab'.format(lang,vocab_type,test_model), fontsize=15, fontweight='bold')
    plt.xlabel('(Pseudo)age in month', fontsize=15)
    plt.ylabel('Proportion of children/models', fontsize=15)
    
    
    plt.tick_params(axis='both', labelsize=10)
    
    # set legend location to avoid shade other curves
    if test_model == 'accum':
        legend_loc = 'lower right'
    else:
        legend_loc = 'upper left'
        
    plt.legend(loc=legend_loc, bbox_to_anchor=(1, 0.5))
    
    plt.savefig('Final_scores/Figures/freq/' + lang + '_' + vocab_type + '.png',dpi=800)
    plt.show()
    


vocab_type_lst = ['exp']
lang_lst = ['AE', 'BE']
test_set_lst = ['CDI']


for vocab_type in vocab_type_lst:


    for lang in lang_lst:

        # set paths
        model_dir = 'Final_scores/Model_eval/' + lang + '/' + vocab_type + '/'
        human_dir = 'Final_scores/Human_eval/CDI/' + lang + '/' + vocab_type

        plot_by_freq(vocab_type, human_dir, test_set_lst,
                         accum_threshold, exp_threshold)



def plot_by_freq(vocab_type,human_dir,test_set,accum_threshold,exp_threshold,test_model):
    
    '''
    merge the test set fir the ease of comparisons
    '''
    
    sns.set_style('whitegrid')
    for file in os.listdir(human_dir):
        human_frame_all = pd.read_csv(human_dir + '/' + file)
    
    # plot the curves respectively
    # plot human  !!! this is fixed
    for freq in set(human_frame_all['group_original'].tolist()):
        
        if freq.split('[')[0] == 'low':
            style = 'dotted'
        else: 
            style = 'solid'
            
        human_frame = human_frame_all[human_frame_all['group_original'].split('[')[0]==freq]
        human_frame = human_frame.iloc[:, 5:-6]
        
        human_result = load_CDI(human_frame)
        ax = sns.lineplot(x="month", y="Proportion of acquired words", data=human_result, color="Red", linestyle=style,linewidth=3, label=lang + '_' + test_set + ': ' + freq)
        
        if test_model == 'accum':
            # plot accumulator model
            accum_all_freq = pd.read_csv(model_dir + 'accum.csv')
            accum_all = accum_all_freq[accum_all_freq['group']==freq]
            accum_result = load_accum(accum_all,accum_threshold)
            ax = sns.lineplot(x="month", y="Lexical score", data=accum_result, color="Purple", linestyle=style,linewidth=2, label='Accumulator'+ ': ' + freq)
        
        
        
        elif test_model == 'speech':
            # plot speech-LSTM model
            speech_result_all = pd.read_csv(model_dir + 'speech.csv')
            # seelct speech model based on the freq band
            speech_result = speech_result_all[speech_result_all['group_median']==freq]
            ax = sns.lineplot(x="month", y="mean_score", data=speech_result,  color="Blue",linewidth=2,linestyle=style, label='LSTM-speech'+ ': ' + freq)
       
        
       
        elif test_model == 'phones':
            # plot phone-LSTM model
            speech_result_all = pd.read_csv(model_dir + 'phones.csv')
            # seelct speech model based on the freq band
            speech_result = speech_result_all[speech_result_all['group_median']==freq]
            ax = sns.lineplot(x="month", y="mean_score", data=speech_result,  color="Green",linewidth=2,linestyle=style, label='LSTM-phones'+ ': ' + freq)
         
            
         
        elif test_model == 'unprompted':
            target_dir = human_dir.replace('CDI', test_set)
            
            # unprompted generation
            for file in os.listdir(target_dir):
                target_frame = pd.read_csv(target_dir + '/' + file)
                
            # read the generated file
            seq_frame_all = pd.read_csv(model_dir + 'unprompted.csv', index_col=0)
            # get the sub-dataframe by frequency  
            score_frame_all, avg_unprompted_all = load_exp(seq_frame_all,target_frame,True,exp_threshold)   
            
            word_group = target_frame[target_frame['group_original']==freq]['word'].tolist()
            score_frame = score_frame_all.loc[word_group]
            avg_values = score_frame.mean()
            month_list_unprompted = [int(x) for x in score_frame.columns]
            ax = sns.lineplot(month_list_unprompted, avg_values.values, color="Green",linestyle=style, linewidth=2, label='LSTM-unprompted'+ ': ' + freq)
       
        
       
        elif test_model == 'prompted':
            target_dir = human_dir.replace('CDI', test_set)
            
            # unprompted generation
            for file in os.listdir(target_dir):
                target_frame = pd.read_csv(target_dir + '/' + file)
                
            # read the generated file
            seq_frame_all = pd.read_csv(model_dir + 'prompted.csv', index_col=0)
            # get the sub-dataframe by frequency  
            score_frame_all, avg_unprompted_all = load_exp(seq_frame_all,target_frame,True,exp_threshold)   
            
            word_group = target_frame[target_frame['group_original']==freq]['word'].tolist()
            score_frame = score_frame_all.loc[word_group]
            avg_values = score_frame.mean()
            month_list_unprompted = [int(x) for x in score_frame.columns]
            ax = sns.lineplot(month_list_unprompted, avg_values.values, color="Blue",linestyle=style, linewidth=2, label='LSTM-prompted'+ ': ' + freq)
        
    # set the limits of the x-axis for each line
    for line in ax.lines:
        plt.xlim(0,36)
        plt.ylim(0,1)
    
    plt.title('{} vocab (tested on {} set)'.format(vocab_type,test_set), fontsize=15, fontweight='bold')
    plt.xlabel('(Pseudo)age in month', fontsize=15)
    plt.ylabel('Proportion of children/models', fontsize=15)
    
    
    plt.tick_params(axis='both', labelsize=10)
    
    plt.legend(loc='upper right')
    
    plt.savefig('Final_scores/Figures/freq/' + lang + '_' + vocab_type + '_' + test_set + '_' + test_model+ '.png',dpi=800)
    plt.show()
    



