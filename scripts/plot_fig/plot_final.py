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
    

'''
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from plot_final_util import load_CDI,load_accum,load_exp,get_score_CHILDES,fit_curve
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
        CHILDES_freq = pd.read_csv('Final_scores/Model_eval/' + lang + '/exp/CDI/CHILDES.csv')
        
        
        
        
        
        avg_values_lst = []
        # averaged by different groups
        for freq in set(list(CHILDES_freq['group_original'].tolist())):
            
            word_group = CHILDES_freq[CHILDES_freq['group_original']==freq]
            score_frame,avg_value = get_score_CHILDES(word_group, exp_threshold)
            avg_values_lst.append(avg_value.values)
        
        
        arrays_matrix = np.array(avg_values_lst)

        # Calculate the average array along axis 0
        avg_values = np.mean(arrays_matrix, axis=0)

        # Plotting the line curve
        month_list_CHILDES = [int(x) for x in score_frame.columns]
        ax = sns.lineplot(month_list_CHILDES, avg_values,color="Orange", linewidth=3, label= 'CHILDES-estimation')
        
        
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
lang = 'BE'    # AE or BE
test_set = 'matched'   # CDI or Wuggy_filtered
exp_threshold = 60
accum_threshold = 200

vocab_type_lst = ['exp']
lang_lst = ['BE','AE']
test_set_lst = ['matched']

for vocab_type in vocab_type_lst:
    for lang in lang_lst:
        for test_set in test_set_lst:

            # set paths
            model_dir = 'Final_scores/Model_eval/' + lang + \
                '/' + vocab_type + '/' + test_set + '/'
            human_dir = 'Final_scores/Human_eval/CDI/' + lang + '/' + vocab_type

            
            plot_all(vocab_type, human_dir, test_set,
                     accum_threshold, exp_threshold)





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
            
            CHILDES_frame = pd.read_csv(model_dir + test_set_lst[n] +  '/CHILDES.csv')
            
            
            
            # unprompted generations
            for freq in freq_lst:   
                
                ref_path = human_dir
                
                if freq.split('[')[0] == 'low':
                    style = 'dotted'
                else: 
                    style = 'solid'
                
                
                
                # CHILDES 
                target_dir = human_dir.replace('CDI', test_set)

                # add human-estimation here
                
                CHILDES_freq = CHILDES_frame[CHILDES_frame['group_original']==freq]
                CHIDES_result, avg_values = get_score_CHILDES(CHILDES_freq, 200)
                month_list_CHILDES = [int(x) for x in CHIDES_result.columns]
                ax = sns.lineplot(month_list_CHILDES, avg_values,
                                  color="Orange",linestyle=style, linewidth=3, label='CHILDES: '+ freq.split('[')[0])
                
                
                
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
exp_threshold = 200


vocab_type_lst = ['exp']
lang_lst = ['AE','BE']
test_set_lst = ['matched']

for vocab_type in vocab_type_lst:
    for lang in lang_lst:
        for test_set in test_set_lst:

            # set paths
            model_dir = 'Final_scores/Model_eval/' + lang + \
                '/' + vocab_type + '/'
            human_dir = 'Final_scores/Human_eval/CDI/' + lang + '/' + vocab_type 

            plot_by_freq(vocab_type,human_dir,test_set_lst,accum_threshold,exp_threshold)



def plot_cal(vocab_type, human_dir, test_set, exp_threshold):
    '''
    plot with extrapolations by frequency
    only take the data after 5 months (starting from 800h,i.e. 9 months)
    find the numeric number for the best fit given the curve
    '''

    sns.set_style('whitegrid')

    # if receptive vocab, load speech-based and phones-based models  # Sample data points
    
    # plot speech-based model 
    if vocab_type == 'recep':

        speech_frame = pd.read_csv(model_dir + 'speech.csv')
        speech_frame_selected = speech_frame[speech_frame['month']>4]
        speech_result = speech_frame_selected.groupby('month')['mean_score'].mean()
        fit_curve(speech_result.index.tolist(), speech_result.tolist(), 0.8, 'Blue', 'speech')
        # plot speech-based model
        phones_frame = pd.read_csv(model_dir + 'phones.csv')
        phones_frame_selected = phones_frame[phones_frame['month']>4]
        phones_result = phones_frame_selected.groupby('month')['mean_score'].mean()
        fit_curve(phones_result.index.tolist(),phones_result.tolist(),  0.8, 'Purple', 'phones')

    elif vocab_type == 'exp':

        # group by different frequencies

        target_dir = human_dir.replace('CDI', test_set)

        # unprompted generation
        for file in os.listdir(target_dir):
            target_frame = pd.read_csv(target_dir + '/' + file)

        seq_frame_all = pd.read_csv(model_dir + 'unprompted.csv', index_col=0)
        score_frame_unprompted, avg_unprompted = load_exp(
            seq_frame_all, target_frame, False, exp_threshold)
        month_list_unprompted = [int(x)
                                 for x in score_frame_unprompted.columns]
        fit_curve(month_list_unprompted, avg_unprompted,  0.8, 'Grey', 'character')
    # set the limits of the x-axis for each line

    plt.title('{} {} vocab'.format(
        lang, vocab_type), fontsize=15, fontweight='bold')
    plt.xlabel('Pseudo age in month(Log scale)', fontsize=15)
    plt.ylabel('Proportion of children/models', fontsize=15)

    plt.tick_params(axis='both', labelsize=10)

    plt.legend()

    plt.grid(True)
    plt.xscale('log')  # Set x-axis to log scale
    plt.yscale('log')
    legend_loc = 'upper left'

    plt.legend(loc=legend_loc)

    plt.savefig('Final_scores/Figures/extrapolation/' + lang + '_' +
                vocab_type + '_extra.png', dpi=800)
    plt.show()


# receptive vocab
vocab_type = 'recep'    # recep or exp
lang = 'BE'    # AE or BE
test_set = 'CDI'   # CDI or Wuggy_filtered
exp_threshold = 200

vocab_type_lst = ['recep', 'exp']
lang_lst = ['AE', 'BE']
test_set_lst = ['CDI']

vocab_type_lst = ['recep']
lang_lst = ['AE', 'BE']
test_set_lst = ['CDI']

for vocab_type in vocab_type_lst:
    for lang in lang_lst:
        for test_set in test_set_lst:

            # set paths
            model_dir = 'Final_scores/Model_eval/' + lang + \
                '/' + vocab_type + '/' + test_set + '/'
            human_dir = 'Final_scores/Human_eval/CDI/' + lang + '/' + vocab_type

            plot_cal(vocab_type, human_dir, test_set, exp_threshold)




def plot_cal_freq(vocab_type,human_dir,model_dir,exp_threshold):
    '''
    plot with extrapolations by frequency
    
    find the numeric number for the best fit given the curve
    '''

    sns.set_style('whitegrid')

    # if receptive vocab, load speech-based and phones-based models  # Sample data points
    
    
    if vocab_type == 'recep':
        
        speech_all_freq_temp = pd.read_csv(model_dir +'speech.csv')
        speech_all_freq = speech_all_freq_temp[speech_all_freq_temp['month']>5]
        freq_set = set(speech_all_freq['group_median'].tolist())
        freq_lst = [[string for string in freq_set if 'high' in string][0],[string for string in freq_set if 'low' in string][0]]
        for freq in freq_lst:   
            
            if freq.split('[')[0] == 'low':
                speech_color = '#008B8B'
                phone_color = '#FF00FF'
            else: 
                speech_color = 'Blue'
                phone_color = 'Purple'
                
            # plot by freq
            speech_frame = speech_all_freq[speech_all_freq['group_median']==freq]
            
            print(speech_frame)
            speech_result = speech_frame.groupby('month')['mean_score'].mean()
            fit_curve(speech_result.index.tolist(), speech_result.tolist(), 0.8, speech_color, 'speech_' + freq.split('[')[0])
            
            # plot speech-based model
            phones_frame_all = pd.read_csv(model_dir + 'phones.csv')
            phones_frame = phones_frame_all[phones_frame_all['month']>5]
            phones_result = phones_frame.groupby('month')['mean_score'].mean()
            fit_curve(phones_result.index.tolist(),phones_result.tolist(),  0.8, phone_color, 'phones_'+freq.split('[')[0])
            print(phones_frame)
            
            
    elif vocab_type == 'exp':

        # group by different frequencies

        target_dir = human_dir.replace('CDI', test_set)

        # unprompted generation
        for file in os.listdir(target_dir):
            target_frame = pd.read_csv(target_dir + '/' + file)
        
        speech_all_freq_temp = pd.read_csv(model_dir +'speech.csv')
        speech_all_freq = speech_all_freq_temp[speech_all_freq_temp['month']>5]
        freq_set = set(speech_all_freq['group_median'].tolist())
        freq_lst = [[string for string in freq_set if 'high' in string][0],[string for string in freq_set if 'low' in string][0]]
        for freq in freq_lst:   
            
            if freq.split('[')[0] == 'low':
                speech_color = '#008B8B'
                
            else: 
                speech_color = 'Grey'
                
                
                
            seq_frame_all = pd.read_csv(model_dir + 'unprompted.csv', index_col=0)
            score_frame_unprompted, avg_unprompted = load_exp(
                seq_frame_all, target_frame, False, exp_threshold)
            month_list_unprompted = [int(x)
                                     for x in score_frame_unprompted.columns]
            fit_curve(month_list_unprompted, avg_unprompted,  0.8, 'Grey', 'character',freq.split('[')[0])
    # set the limits of the x-axis for each line

    plt.title('{} {} vocab'.format(
        lang, vocab_type), fontsize=15, fontweight='bold')
    plt.xlabel('Pseudo age in month(Log scale)', fontsize=15)
    plt.ylabel('Proportion of children/models', fontsize=15)

    plt.tick_params(axis='both', labelsize=10)

    plt.legend()

    plt.grid(True)
    plt.xscale('log')  # Set x-axis to log scale
    
    legend_loc = 'upper left'

    plt.legend(fontsize='small',loc=legend_loc)

    plt.savefig('Final_scores/Figures/extrapolation/' + lang + '_' +
                vocab_type + '_extra_freq.png', dpi=800)
    plt.show()
    
    
vocab_type = 'recep'    # recep or exp
lang = 'BE'    # AE or BE
test_set = 'CDI'   # CDI or Wuggy_filtered
exp_threshold = 200

vocab_type_lst = ['recep', 'exp']
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

            plot_cal_freq(vocab_type, human_dir, model_dir,exp_threshold)
