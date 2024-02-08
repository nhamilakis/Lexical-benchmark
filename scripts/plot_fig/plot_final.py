#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jliu

plot the result figs

All: all sets in one fig
by_freq: plot the given set


- original (Q:necessary to do so?  preserve anyway)
- extrapolation

"""
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from plot_final_util import load_CDI,load_accum,load_exp,get_score_CHILDES,fit_sigmoid,fit_log,get_linear
import numpy as np
import sys
import argparse

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Plot all figures in the paper')

    parser.add_argument('--lang_lst', default=['AE','BE'],
                        help='languages to test: AE, BE or FR')

    parser.add_argument('--vocab_type_lst', default=['exp'],
                        help='which type of words to evaluate; recep or exp')

    parser.add_argument('--test_lst', default=['6_audiobook_aligned'],
                        help='which testset to evaluate')

    parser.add_argument('--eval_dir', type=str, default='Final_scores/Model_eval/',
                        help='directory of machine CDI results')

    parser.add_argument('--test_dir', type=str, default="/data/Machine_CDI/Lexical-benchmark_output/"
                                                         "test_set/matched_set/char/bin_range_aligned/",
                        help='directory of test set')

    parser.add_argument('--fig_dir', type=str, default="Final_scores/Figures/",
                        help='directory of human CDI')

    parser.add_argument('--exp_threshold', type=int, default=60,
                        help='threshold for expressive vocab')

    parser.add_argument('--accum_threshold', type=int, default=200,
                        help='thresho, offsetld for accumulator model')
    
    parser.add_argument('--color_dict', type=int, default={'speech': 'Blue','phones':'Green',
                                                'human':'Red','CHILDES':'Orange','unprompted':'Grey'},
                        help='color settings of differnt test sets')

    parser.add_argument('--by_freq', default=False,
                        help='whether to decompose the results by frequency bands')

    parser.add_argument('--extrapolation', default=False,
                        help='whether to plot the extrapolation figure')
    
    parser.add_argument('--aggregate_freq', default=False,
                        help='whether to aggregate frequency bands into 2')
    
    parser.add_argument('--set_name', default='model',
                        help='different sets in a more fine-grained frequency bands: human, model, CHILDES')
    
    parser.add_argument('--target_y', type=float, default=0.8,
                        help='target y to reach for extrapolation')
    
    return parser.parse_args(argv)

sns.set_style('whitegrid')



test_dir = '/data/Machine_CDI/Lexical-benchmark_output/test_set/matched_set/char/bin_range_aligned/6_audiobook_aligned/'

eval_dir = '/data/Machine_CDI/Lexical-benchmark_output/Final_scores/Model_eval/BE/exp/'

lang = 'BE'

vocab_type = 'exp'

color_dict = {'speech': 'Blue','phones':'Green','human':'Red','CHILDES':'Orange','unprompted':'Grey'}

exp_threshold = 1

def load_results(test_dir,eval_dir,lang,vocab_type,set_name,color_dict,exp_threshold):

    """
    load data for plotting
    input:
        - testset dir: human CDI and matched CDI
        - eval dir: eval results
        - vocab_type: exp or recep or individual set
        - set_name: all or specific name
    return
        dict {SetName: [dataframe,color]}
    """
    
    def load_data(vocab_type,set_name,eval_dir,test_dir,exp_threshold):
        
        '''
        load data based on vocab type and set name
        '''
        data_dict = {}
        # load human
        if set_name == 'human':
            human_frame = pd.read_csv(test_dir + 'human_' + lang + '_' + vocab_type +'.csv')
            test_frame,test_x,test_y = load_CDI(human_frame)
            
        elif vocab_type == 'exp':
            data_frame = pd.read_csv(eval_dir + set_name + '.csv')
            target_frame = pd.read_csv(test_dir + 'machine_' + lang + '_' + vocab_type +'.csv')
            test_frame,test_x,test_y = load_exp(data_frame,target_frame,exp_threshold)
        
        data_dict[set_name] = [test_frame,test_x,test_y,color_dict[set_name]]
        return data_dict
    
    
    if vocab_type == 'recep':
        pass
    elif vocab_type == 'exp':
        # load human CDI  
        human_dict = load_data(vocab_type,'human',eval_dir,test_dir,exp_threshold)
        # load CHILDES
        #CHILDES_dict = load_data(vocab_type,'CHILDES',eval_dir,test_dir,exp_threshold)
        # load unprompted generation
        unprompted_dict = load_data(vocab_type,'unprompted',eval_dir,test_dir,exp_threshold)
        # load prompted generation
        #prompted_dict = load_data(vocab_type,'prompted',eval_dir,test_dir,exp_threshold)
        #final_dict = {**human_dict,**CHILDES_dict,**unprompted_dict}
        
        final_dict = {**human_dict,**unprompted_dict}
    # load the testset name to plot by freq
    # elif vocab_type == 'individual':
       
    return final_dict




def plot_all(final_dict):

    """
    input: dataframes of input data to print

    """
    
    
    for label, data in final_dict.items():
        
        # plot the curve recursively 
        sns.lineplot(data[1], data[2],linewidth=3, label=label,color = data[3])
    # set the limits of the x-axis for each line
    
    plt.xlim(0, 36)
    plt.ylim(0, 1)

    plt.title('{} {} vocab'.format(
        lang, vocab_type, test_set), fontsize=15, fontweight='bold')
    plt.xlabel('(Pseudo) age in month', fontsize=15)
    plt.ylabel('Proportion of children/models', fontsize=15)

    plt.tick_params(axis='both', labelsize=10)

    
    legend_loc = 'upper left'

    plt.legend(loc=legend_loc)

    plt.savefig('Final_scores/Figures/avg/' + lang + '_' +
                vocab_type + '_' + test_set+'.png', dpi=800)
    plt.show()



def plot_by_freq(vocab_type,human_dir,model_dir,test_set,accum_threshold,exp_threshold
                 ,lang,set_type,extrapolation,target_y):
    
    '''
    plot mdifferent freq bands in one figure
    '''
    
    sns.set_style('whitegrid')
    
    # re-annotate the freq bands based on aggregation num

   
    for file in os.listdir(human_dir):
        human_frame_all = pd.read_csv(human_dir + '/' + file)
        
        freq_lst = set(human_frame_all['group'].tolist())
        
    
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
                    if not extrapolation:
                        ax = sns.lineplot(month_list_CHILDES, avg_values,
                                      linewidth=3, label=freq)
                    if extrapolation:
                        fit_sigmoid(month_list_CHILDES, avg_values, target_y,0, 'CHILDES','solid')
                        
                if set_type == 'model':
                    # unprompted generation
                    if test_type == 'aligned':
                        target_frame = pd.read_csv(target_dir + '/median_aligned.csv' )
                        
                    else:
                        if test_type == 'unaligned':
                            target_frame = pd.read_csv(target_dir + '/median_unaligned.csv' )
                        
                    # read the generated file
                    seq_frame_all = pd.read_csv(model_dir + 'unprompted.csv', index_col=0)
                    # get the sub-dataframe by frequency  
                    score_frame_all, avg_unprompted_all = load_exp(seq_frame_all,target_frame,True,exp_threshold)   
                    
                    word_group = target_frame[target_frame['group']==freq]['word'].tolist()
                    score_frame = score_frame_all.loc[word_group]
                    avg_values = score_frame.mean()
                    month_list_unprompted = [int(x) for x in score_frame.columns]
                    if not extrapolation:
                        ax = sns.lineplot(month_list_unprompted, avg_values.values, linewidth=3, label= freq)
                    if extrapolation:
                        fit_sigmoid(month_list_unprompted, avg_values.values,target_y, 0,str(freq),'solid')
                        
        
      
    # set the limits of the x-axis for each line
    if not extrapolation:
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
    
    


'''
lang = 'AE'
vocab_type = 'exp'
final_frame = pd.read_csv('Final_scores/Figures/' + lang + '_' + vocab_type + '_extra.csv')
    
final_frame_grouped =final_frame.groupby('Type')
for testset, final_frame_group in final_frame_grouped:
    sns.lineplot(x="Freq", y="estimated_month", data=final_frame_group, linewidth=2,label= testset)
plt.title(lang + ' extrapolated months', fontsize=15, fontweight='bold')

'''



def main(argv):
    # Args parser
    args = parseArgs(argv)
    accum_threshold = args.accum_threshold
    exp_threshold = args.exp_threshold
    color_dict = args.color_dict
    
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
                                 accum_threshold,exp_threshold,lang,args.set_type
                                 ,args.extrapolation,args.target_y)
                            
                            
                        else:
                            aggre_freq(vocab_type,human_dir,model_dir,test_set,accum_threshold,exp_threshold
                                               ,lang,False,args.target_y)
                        
                else:
                    if not args.by_freq:
                        print('plotting cal')
                        plot_cal(vocab_type, human_dir, model_dir, test_set, exp_threshold,lang,args.target_y)
                    
                    if args.aggregate_freq:
                        
                       aggre_freq(vocab_type,human_dir,model_dir,test_set,accum_threshold,exp_threshold
                                      ,lang,args.extrapolation,args.target_y)
                       
                    else:
                        plot_by_freq(vocab_type,human_dir,model_dir,test_set,
                              accum_threshold,exp_threshold,lang,args.set_type
                              ,args.extrapolation,args.target_y)
                        
                      
                '''   
                plot_extra_curve(vocab_type,human_dir,model_dir,test_set,accum_threshold,exp_threshold
                                 ,lang,args.target_y)  
                
                '''
                        
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)