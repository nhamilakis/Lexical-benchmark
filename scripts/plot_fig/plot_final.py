#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: jliu

color setting
human: red
freq-based:purpl -> pink for matched
model:
    1.speech/prompted (the most human like): speech
    2.phone: blue
    2.phoneme/unprompted: purple
by_freq: similar colors but different shape
    high: line
    low: dotted
    
/data/Lexical-benchmark_output/test_set/matched_set/char/bin_range_aligned/6
'''
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from plot_final_util import load_CDI,load_accum,load_exp,get_score_CHILDES,fit_sigmoid,fit_log
import numpy as np
import sys
import argparse

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Plot all figures in the paper')

    parser.add_argument('--lang_lst', default=['AE','BE'],
                        help='languages to test: AE, BE or FR')

    parser.add_argument('--vocab_type_lst', default=['recep'],
                        help='which type of words to evaluate; recep or exp')

    parser.add_argument('--testset_lst', default=['matched'],
                        help='which testset to evaluate')

    parser.add_argument('--model_dir', type=str, default='Final_scores/Model_eval/',
                        help='directory of machine CDI results')

    parser.add_argument('--human_dir', type=str, default="Final_scores/Human_eval/CDI/",
                        help='directory of human CDI')

    parser.add_argument('--fig_dir', type=str, default="Final_scores/Figures/",
                        help='directory of human CDI')

    parser.add_argument('--exp_threshold', type=int, default=60,
                        help='threshold for expressive vocab')

    parser.add_argument('--accum_threshold', type=int, default=200,
                        help='thresho, offsetld for accumulator model')

    parser.add_argument('--by_freq', default=False,
                        help='whether to decompose the results by frequency bands')

    parser.add_argument('--extrapolation', default=False,
                        help='whether to plot the extrapolation figure')
    
    parser.add_argument('--aggregate_freq', default=False,
                        help='whether to aggregate frequency bands into 2')
    
    parser.add_argument('--set_type', default='speech',
                        help='different sets in a more fine-grained frequency bands: human, model, CHILDES')
    
    parser.add_argument('--target_y', type=float, default=0.8,
                        help='target y to reach for extrapolation')
    
    return parser.parse_args(argv)

sns.set_style('whitegrid')

def plot_all(vocab_type, human_dir,model_dir, test_set, accum_threshold, exp_threshold,lang):

    # plot the curve averaged by freq bands

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

    # load speech-based and phones-based models
    if vocab_type == 'recep':

        # plot accumulator model
        accum_all = pd.read_csv(model_dir + 'accum.csv')
        accum_result = load_accum(accum_all, accum_threshold)
        ax = sns.lineplot(x="month", y="Lexical score", data=accum_result,
                          color="Green", linewidth=3, label='Accumulator')

        speech_result = pd.read_csv(model_dir + 'speech.csv')
        
        #speech_result = pd.read_csv('/data/recep_results/speech/0.5_0.8/matched_BE_recep.csv')
        # remove the chunk in which there is only one word in each freq band
        # merge among similar freq band of the same chunk
        speech_result = speech_result.groupby(['month','chunk'])['mean_score'].mean().reset_index()
        ax = sns.lineplot(x="month", y="mean_score", data=speech_result,       
                          color='Blue', linewidth=1, label='LSTM-speech')     # show the freq band's level range 
        
        
        # plot speech-based model
        phones_result = pd.read_csv(model_dir + 'phones.csv')
        phones_result = phones_result.groupby(['month','chunk'])['mean_score'].mean().reset_index()
        ax = sns.lineplot(x="month", y="mean_score", data=phones_result,
                          color="Purple", linewidth=2, label='LSTM-phones')

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
        
        
        # unprompted generation: different freq bands
        for file in os.listdir(target_dir):
            target_frame = pd.read_csv(target_dir + '/' + file)

        seq_frame_all = pd.read_csv(model_dir + 'unprompted.csv', index_col=0)
        score_frame_unprompted, avg_unprompted = load_exp(
            seq_frame_all, target_frame, False, exp_threshold)
        month_list_unprompted = [int(x)
                                 for x in score_frame_unprompted.columns]
        ax = sns.lineplot(month_list_unprompted, avg_unprompted,
                          color="Grey", linewidth=3, label='LSTM-unprompted')


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



def plot_by_freq(vocab_type,human_dir,model_dir,test_set,accum_threshold,exp_threshold,lang,set_type):
    
    '''
    plot mdifferent freq bands in one figure
    '''
    
    sns.set_style('whitegrid')
    
    # re-annotate the freq bands based on aggregation num

   
    for file in os.listdir(human_dir):
        human_frame_all = pd.read_csv(human_dir + '/' + file)
        
        freq_lst = set(human_frame_all['group_original'].tolist())
        
    
    if set_type == 'human':
    
        for freq in freq_lst:
        
            # read the results recursively
            
            human_frame = human_frame_all[human_frame_all['group_original']==freq]
            human_frame = human_frame.iloc[:, 5:-6]
            human_result = load_CDI(human_frame)
            ax = sns.lineplot(x="month", y="Proportion of acquired words", data=human_result, linewidth=3.5, label= freq)
            
    
    if vocab_type == 'recep':
            
            for freq in freq_lst:   
                
                # plot accumulator model
                if set_type == 'accum':
                # accumulator model
                    accum_all_freq = pd.read_csv(model_dir + test_set +'/accum.csv')
                    accum_all = accum_all_freq[accum_all_freq['group']==freq]
                    accum_result = load_accum(accum_all,accum_threshold)
                    ax = sns.lineplot(x="month", y="Lexical score", data=accum_result, color="Green", linewidth=3, label = freq)
                
                if set_type == 'speech':
                    speech_result_all = pd.read_csv(model_dir + 'speech.csv')  
                    # seelct speech model based on the freq band
                    speech_result = speech_result_all[speech_result_all['group']==freq]
                    ax = sns.lineplot(x="month", y="mean_score", data=speech_result, linewidth=3,label= freq)
                    
                if set_type == 'phoneme':   
                    # plot phone-LSTM model
                    phone_result_all = pd.read_csv(model_dir + test_set + '/phones.csv')
                    # seelct speech model based on the freq band
                    phone_result = phone_result_all[phone_result_all['group_median']==freq]
                    ax = sns.lineplot(x="month", y="group", data=phone_result,linewidth=3, label= freq)
                    
                
         
    elif vocab_type == 'exp':
            
            
            # unprompted generations
            for freq in freq_lst:   
                
                # CHILDES 
                target_dir = human_dir.replace('CDI', test_set)
            
                
                if set_type == 'CHILDES':
                    # add human-estimation here
                    CHILDES_frame = pd.read_csv('Final_scores/Model_eval/' + lang + '/exp/CDI/CHILDES.csv')
                    CHILDES_freq = CHILDES_frame[CHILDES_frame['group_original']==freq]
                    CHIDES_result, avg_values = get_score_CHILDES(CHILDES_freq, exp_threshold)
                    month_list_CHILDES = [int(x) for x in CHIDES_result.columns]
                    ax = sns.lineplot(month_list_CHILDES, avg_values,
                                      linewidth=3, label=freq)
                
                
                if set_type == 'model':
                    # unprompted generation
                    for file in os.listdir(target_dir):
                        target_frame = pd.read_csv(target_dir + '/' + file)
                        
                    # read the generated file
                    seq_frame_all = pd.read_csv(model_dir + 'unprompted.csv', index_col=0)
                    # get the sub-dataframe by frequency  
                    score_frame_all, avg_unprompted_all = load_exp(seq_frame_all,target_frame,True,exp_threshold)   
                    
                    word_group = target_frame[target_frame['group']==freq]['word'].tolist()
                    score_frame = score_frame_all.loc[word_group]
                    avg_values = score_frame.mean()
                    month_list_unprompted = [int(x) for x in score_frame.columns]
                    ax = sns.lineplot(month_list_unprompted, avg_values.values, linewidth=3, label= freq)
                    
        
      
    # set the limits of the x-axis for each line
    for line in ax.lines:
        plt.xlim(0,36)
        plt.ylim(0,1)
    
    plt.title('{} {} vocab'.format(lang,vocab_type), fontsize=15, fontweight='bold')
    
    
    if set_type == 'model':
        ylabel = 'Proportion of models'
        xlabel = 'Pseudo age in month'
        
    else:
        ylabel = 'Proportion of children'
        xlabel = 'Age in month'
        
    plt.ylabel(ylabel, fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.tick_params(axis='both')
    
    # set legend location to avoid shade other curves
    
    if vocab_type == 'exp':
        legend_loc = 'upper left'
        # Create proxy artists for the legend labels
        plt.legend(fontsize='small',loc=legend_loc)
        
    else:
        legend_loc = 'upper right'
        plt.legend(fontsize='small', loc=legend_loc)
    plt.legend(title='Freq bands')  
    plt.savefig('Final_scores/Figures/freq/' + lang + '_' + vocab_type + '_' + set_type + '_freq.png',dpi=800)
    plt.show()
    
    

     
def aggre_freq(vocab_type,human_dir,model_dir,test_set,accum_threshold,exp_threshold
               ,lang,extrapolation,target_y):
    
    '''
    plot mdifferent freq bands in one figure
    aggregate into 2 bands: high or low      
     
    '''
    style = ['solid','dotted']
    sns.set_style('whitegrid')
    sub_dict = {0: 'low', 1: 'low',2: 'low', 3: 'high',4: 'high', 5: 'high'}
    freq_lst = ['high', 'low']
    
    if not extrapolation:
        for file in os.listdir(human_dir):
            human_frame_all = pd.read_csv(human_dir + '/' + file)
            human_frame_all['group_original'] = human_frame_all['group_original'].replace(sub_dict)
            
            
            n = 0
            for freq in freq_lst:
            
                # read the results recursively
                human_frame = human_frame_all[human_frame_all['group_original']==freq]
                human_frame = human_frame.iloc[:, 5:-6]
                
                human_result = load_CDI(human_frame)
                ax = sns.lineplot(x="month", y="Proportion of acquired words", data=human_result
                                  , linewidth=3.5, label= 'human: ' + freq, color = 'Red',linestyle = style[n])
                n += 1
    
    if vocab_type == 'recep':
            
        # replace freq group
        # accum_all_freq = pd.read_csv(model_dir + test_set +'/accum.csv')
        
        speech_result_all = pd.read_csv(model_dir + 'speech.csv')  
        #phone_result_all = pd.read_csv(model_dir + test_set + '/phones.csv')
        
        # accum_all_freq['group'] = accum_all_freq['group'].replace(sub_dict)
        speech_result_all['group'] = speech_result_all['group'].replace(sub_dict)
        #phone_result_all = phone_result_all['group'].replace(sub_dict)
        
        n = 0
        for freq in freq_lst:   
                
                '''
                accum_all = accum_all_freq[accum_all_freq['group']==freq]
                accum_result = load_accum(accum_all,accum_threshold)
                ax = sns.lineplot(x="month", y="Lexical score", data=accum_result, color="Green", linewidth=3, label = freq)
                '''
                # seelct speech model based on the freq band
                speech_result = speech_result_all[speech_result_all['group']==freq]
                if extrapolation:
                    fit_log(speech_result.index.tolist(), speech_result.tolist(),target_y, 'Blue', 'speech')
                else:    
                    ax = sns.lineplot(x="month", y="mean_score", data=speech_result, linewidth=3,label= 'speech: ' + freq
                                  ,linestyle = style[n], color = 'Blue')
                  
                # plot phone-LSTM model
                
                # seelct speech model based on the freq band
                '''
                phone_result = phone_result_all[phone_result_all['group']==freq]
                ax = sns.lineplot(x="month", y="group", data=phone_result,linewidth=3, label= freq)
                '''  
                n += 1
         
    elif vocab_type == 'exp':
            
            # add human-estimation here
            CHILDES_frame = pd.read_csv('Final_scores/Model_eval/' + lang + '/exp/CDI/CHILDES.csv')
            # read the generated file
            target_dir = human_dir.replace('CDI', test_set)
            for file in os.listdir(target_dir):
                target_frame = pd.read_csv(target_dir + '/' + file)
            target_frame['group']=target_frame['group'].replace(sub_dict)
            CHILDES_frame['group_original'] = CHILDES_frame['group_original'].replace(sub_dict)
            
            
            n = 0
            # unprompted generations
            for freq in freq_lst:   
                
                # CHILDES 
                
                CHILDES_freq = CHILDES_frame[CHILDES_frame['group_original']==freq]
                CHIDES_result, avg_values = get_score_CHILDES(CHILDES_freq, exp_threshold)
                month_list_CHILDES = [int(x) for x in CHIDES_result.columns]
                if extrapolation:
                    fit_sigmoid(month_list_CHILDES, avg_values,target_y, 5, 'orange','CHILDES',style[n])
                else:    
                    ax = sns.lineplot(month_list_CHILDES, avg_values,linestyle = style[n],
                                      linewidth=3, label= 'CHILDES ' + freq, color = 'Orange')
                
                n += 1 
                
            n = 0
           # unprompted generations
            for freq in freq_lst:      
                # unprompted generation
                   
                seq_frame_all = pd.read_csv(model_dir + 'unprompted.csv', index_col=0)
                # get the sub-dataframe by frequency  
                score_frame_all, avg_unprompted_all = load_exp(seq_frame_all,target_frame,True,exp_threshold)   
                    
                word_group = target_frame[target_frame['group']==freq]['word'].tolist()
                score_frame = score_frame_all.loc[word_group]
                avg_values = score_frame.mean()
                month_list_unprompted = [int(x) for x in score_frame.columns]
                if extrapolation:
                    fit_sigmoid(month_list_unprompted, avg_values,target_y, 5, 'grey','model',style[n])
                else:   
                    ax = sns.lineplot(month_list_unprompted, avg_values.values, linewidth=3
                                  , label= 'Char: ' + freq, color = 'Grey',linestyle = style[n])
                
                n += 1
       
    # set the limits of the x-axis for each line
    if not extrapolation:
        for line in ax.lines:
            plt.xlim(0,36)
            plt.ylim(0,1)
    
    plt.title('{} {} vocab'.format(lang,vocab_type), fontsize=15, fontweight='bold')
    
    ylabel = 'Proportion of children/models'
    xlabel = '(Pseudo) age in month'
    
    plt.ylabel(ylabel, fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.tick_params(axis='both')
    
    # set legend location to avoid shade other curves
    
    if vocab_type == 'exp':
        legend_loc = 'upper left'
        # Create proxy artists for the legend labels
        plt.legend(fontsize='small',loc=legend_loc)
        
    else:
        legend_loc = 'upper right'
        plt.legend(fontsize='small', loc=legend_loc)
    
    plt.savefig('Final_scores/Figures/freq/' + lang + '_' + vocab_type + '_freq.png',dpi=800)
    plt.show()
    
    
  
    
def plot_cal(vocab_type, human_dir, model_dir, test_set, exp_threshold,lang,target_y):
    '''
    plot with extrapolations by frequency
    only take the data after 5 months (starting from 800h,i.e. 9 months)
    find the numeric number for the best fit given the curve
    
    '''

    sns.set_style('whitegrid')


    # plot speech-based model
    if vocab_type == 'recep':
        # plot speech-based model
        speech_frame = pd.read_csv(model_dir + 'speech.csv')
        speech_frame_selected = speech_frame[speech_frame['month'] > 4]
        speech_result = speech_frame_selected.groupby('month')['mean_score'].mean()
        #fit_log(speech_result.index.tolist(), speech_result.tolist(),target_y, 'Blue', 'speech')
        fit_sigmoid(speech_result.index.tolist(), speech_result.tolist(),target_y, 0,'Blue', 'speech')
        '''
        phones_frame = pd.read_csv(model_dir + 'phones.csv')
        phones_frame_selected = phones_frame[phones_frame['month'] > 4]
        phones_result = phones_frame_selected.groupby('month')['mean_score'].mean()
        fit_log(phones_result.index.tolist(), phones_result.tolist(),target_y, 'Purple', 'phones')
        '''
    elif vocab_type == 'exp':

        # group by different frequencies

        target_dir = human_dir.replace('CDI', test_set)

        # add human-estimation here
        CHILDES_freq = pd.read_csv('Final_scores/Model_eval/' + lang + '/exp/CDI/CHILDES.csv')

        avg_values_lst = []
        # averaged by different groups
        for freq in set(list(CHILDES_freq['group_original'].tolist())):
            word_group = CHILDES_freq[CHILDES_freq['group_original'] == freq]
            score_frame, avg_value = get_score_CHILDES(word_group, exp_threshold)
            avg_values_lst.append(avg_value.values)

        arrays_matrix = np.array(avg_values_lst)

        # Calculate the average array along axis 0
        avg_values = np.mean(arrays_matrix, axis=0)

        # Plotting the line curve
        month_list_CHILDES = [int(x) for x in score_frame.columns]


        para_dict = fit_sigmoid(month_list_CHILDES, avg_values,target_y, 5, 'orange','CHILDES','solid')
        

        # unprompted generation
        for file in os.listdir(target_dir):
            target_frame = pd.read_csv(target_dir + '/' + file)

        seq_frame_all = pd.read_csv(model_dir + 'unprompted.csv', index_col=0)
        score_frame_unprompted, avg_unprompted = load_exp(
            seq_frame_all, target_frame, False, exp_threshold)
        month_list_unprompted = [int(x)
                                 for x in score_frame_unprompted.columns]

        para_dict = fit_sigmoid(month_list_unprompted, avg_unprompted,target_y, 5, 'grey','model','solid')
      
    # set the limits of the x-axis for each line
    plt.title('{} {} vocab'.format(
        lang, vocab_type), fontsize=15, fontweight='bold')
    plt.xlabel('Pseudo age in month', fontsize=15)
    plt.ylabel('Proportion of children/models', fontsize=15)

    plt.tick_params(axis='both', labelsize=10)

    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True)

    legend_loc = 'upper left'

    plt.legend(loc=legend_loc)

    fig_dir = 'Final_scores/Figures/extrapolation/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    plt.savefig(fig_dir + lang + '_' +
                vocab_type + '_extra.png', dpi=800)
    plt.show()




def plot_cal_freq(vocab_type, human_dir, model_dir, exp_threshold, test_set, lang):
    '''
    plot with extrapolations by frequency

    find the numeric number for the best fit given the curve
    '''

    sns.set_style('whitegrid')

    # if receptive vocab, load speech-based and phones-based models  # Sample data points

    if vocab_type == 'recep':

        speech_all_freq_temp = pd.read_csv(model_dir + 'speech.csv')
        speech_all_freq = speech_all_freq_temp[speech_all_freq_temp['month'] > 5]
        freq_set = set(speech_all_freq['group_median'].tolist())
        freq_lst = [[string for string in freq_set if 'high' in string][0],
                    [string for string in freq_set if 'low' in string][0]]
        for freq in freq_lst:

            if freq.split('[')[0] == 'low':
                speech_color = '#008B8B'
                phone_color = '#FF00FF'
            else:
                speech_color = 'Blue'
                phone_color = 'Purple'

            # plot by freq
            speech_frame = speech_all_freq[speech_all_freq['group_median'] == freq]

            print(speech_frame)
            speech_result = speech_frame.groupby('month')['mean_score'].mean()
            fit_curve(speech_result.index.tolist(), speech_result.tolist(), 0.8, speech_color,
                      'speech_' + freq.split('[')[0])

            # plot speech-based model
            phones_frame_all = pd.read_csv(model_dir + 'phones.csv')
            phones_frame = phones_frame_all[phones_frame_all['month'] > 5]
            phones_result = phones_frame.groupby('month')['mean_score'].mean()
            fit_curve(phones_result.index.tolist(), phones_result.tolist(), 0.8, phone_color,
                      'phones_' + freq.split('[')[0])
            print(phones_frame)


    elif vocab_type == 'exp':

        # group by different frequencies

        target_dir = human_dir.replace('CDI', test_set)

        # unprompted generation
        for file in os.listdir(target_dir):
            target_frame = pd.read_csv(target_dir + '/' + file)

        seq_frame_all = pd.read_csv(model_dir + 'unprompted.csv', index_col=0)

        for file in os.listdir(human_dir):
            human_frame_all = pd.read_csv(human_dir + '/' + file)

            freq_lst = set(human_frame_all['group_original'].tolist())

        for freq in freq_lst:
            word_group = target_frame[target_frame['group'] == freq]['word'].tolist()
            score_frame = seq_frame_all.loc[word_group]

            score_frame_unprompted, avg_unprompted = load_exp(
                score_frame, target_frame, False, exp_threshold)
            month_list_unprompted = [int(x)
                                     for x in score_frame_unprompted.columns]

            x_data, y_data, x_fit, y_fit, para_dict = fit_sigmoid(month_list_unprompted, avg_unprompted, 0.8, 5)
            plt.scatter(x_data, y_data)
            plt.plot(x_fit, y_fit, linewidth=3, label=str(freq))
    # set the limits of the x-axis for each line

    plt.title('{} {} vocab'.format(
        lang, vocab_type), fontsize=15, fontweight='bold')
    plt.xlabel('Pseudo age in month(Log scale)', fontsize=15)
    plt.ylabel('Proportion of children/models', fontsize=15)

    plt.tick_params(axis='both', labelsize=10)

    plt.legend()

    plt.grid(True)

    legend_loc = 'upper left'

    plt.legend(fontsize='small', loc=legend_loc)

    plt.savefig('Final_scores/Figures/extrapolation/' + lang + '_' +
                vocab_type + '_extra_freq.png', dpi=800)
    plt.show()


   


def main(argv):
    # Args parser
    args = parseArgs(argv)
    accum_threshold = args.accum_threshold
    exp_threshold = args.exp_threshold
    
    for vocab_type in args.vocab_type_lst:
        for lang in args.lang_lst:
            for test_set in args.testset_lst:

                model_dir = args.model_dir + lang + \
                            '/' + vocab_type + '/' + test_set + '/'
                human_dir = args.human_dir + lang + '/' + vocab_type
                
                
                
                if not args.extrapolation:
                    
                    if not args.by_freq:
                        plot_all(vocab_type, human_dir,model_dir, test_set,
                                 accum_threshold, exp_threshold, lang)
                    else:
                        if not args.aggregate_freq:
                            plot_by_freq(vocab_type,human_dir,model_dir,test_set,
                                 accum_threshold,exp_threshold,lang,args.set_type)
                            
                        else:
                            aggre_freq(vocab_type,human_dir,model_dir,test_set,accum_threshold,exp_threshold
                                               ,lang,False,args.target_y)
                        
                else:
                    if not args.by_freq:
                        print('plotting cal')
                        plot_cal(vocab_type, human_dir, model_dir, test_set, exp_threshold,lang,args.target_y)
                        
                    else:
                       
                        aggre_freq(vocab_type,human_dir,model_dir,test_set,accum_threshold,exp_threshold
                                           ,lang,True,args.target_y)
                            

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)