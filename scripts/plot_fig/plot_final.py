#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot all figures in the paper

@author: jliu
"""
'''
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
'''
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from plot_final_util import load_CDI,load_accum,load_exp,get_score_CHILDES,fit_curve
import numpy as np
import sys
import argparse

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Investigate CHILDES corpus')

    parser.add_argument('--lang_lst', default=['AE','BE'],
                        help='languages to test: AE, BE or FR')

    parser.add_argument('--vocab_type_lst', default=['recep','exp'],
                        help='which type of words to evaluate; recep or exp')

    parser.add_argument('--testset_lst', default=['CDI', 'matched'],
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
                        help='threshold for accumulator model')

    parser.add_argument('--by_freq', default=False,
                        help='whether to decompose the results by frequency bands')

    parser.add_argument('--extrapolation', default=False,
                        help='whether to plot the extrapolation figure')

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




def main(argv):
    # Args parser
    args = parseArgs(argv)

    for vocab_type in args.vocab_type_lst:
        for lang in args.lang_lst:
            for test_set in args.test_set_lst:

                model_dir = args.model_dir + lang + \
                            '/' + vocab_type + '/' + test_set + '/'
                human_dir = args.human_dir + lang + '/' + vocab_type

                plot_all(vocab_type, human_dir, test_set,
                         accum_threshold, exp_threshold)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)