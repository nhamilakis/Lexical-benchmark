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
from typing import Any
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
from stat_util import get_freq_table, match_range, get_equal_bins, match_bin_range, plot_density_hist, \
    match_bin_density, match_bin_prop, select_type
from aochildes.dataset import AOChildesDataSet  # !!! this should be configured later
import math


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Select test sets by freq')

    parser.add_argument('--lang', type=str, default='BE',
                        help='languages to test: AE, BE or FR')

    parser.add_argument('--eval_condition', type=str, default='recep',
                        help='which type of words to select; recep or exp')

    parser.add_argument('--dataPath', type=str, default='/data/Lexical-benchmark_data/test_set/',
                        help='path to the freq corpus')

    parser.add_argument('--outPath', type=str, default='/data/Lexical-benchmark_output/test_set/',
                        help='output path to the freq corpus')

    parser.add_argument('--word_format', type=str, default='char',
                        help='char or phon format')

    parser.add_argument('--num_bins', type=int, default=6,
                        help='number of eaqul-sized bins of human CDI data')

    parser.add_argument('--freq_type', type=str, default='freq',
                        help='freq types: freq or log_freq')

    parser.add_argument('--machine_set', type=str, default='wuggy',
                        help='different sets of machine CDI: wuggy or audiobook')

    parser.add_argument('--match_mode', type=str, default='bin_range_aligned',
                        help='different match modes: range_aligned, bin_range_aligned ,density_aligned or distr_aligned')

    parser.add_argument('--threshold', type=float, default=0.05,
                        help='difference threshold of each bin')  # for later usage

    parser.add_argument('--word_type', type=str, default='content',
                        help='difference word types')  # for later usage

    return parser.parse_args(argv)


def get_freq_frame(test, train_path, word_type):
    """
    get the overall freq of CHILDES and train data fre
    """

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

    # !!! TO DO: add preprocessing steps of CDI tests sets

    # Find the intersection of the three lists
    selected_words = list(set(test['word'].tolist()).intersection(set(CHILDES_fre_table['Word'].tolist()),
                                                                  set(audiobook_fre_table['Word'].tolist()),
                                                                  set(CELEX['Word'].tolist())))

    # get the dataframe with all the dataset
    freq_frame = pd.DataFrame([selected_words]).T
    freq_frame.rename(columns={0: 'word'}, inplace=True)
    freq_frame= freq_frame.dropna()
    freq_frame= select_type(freq_frame, word_type)
    selected_words = list(set(freq_frame['word'].tolist()))

    audiobook_lst = []
    CELEX_lst = []
    CHILDES_lst = []

    for word in selected_words:
        try:
            audiobook_lst.append(
                audiobook_fre_table[audiobook_fre_table['Word'] == word]['Norm_freq_per_million'].item())
            CELEX_lst.append(CELEX[CELEX['Word'] == word]['SUBTLWF'].item())
            CHILDES_lst.append(CHILDES_fre_table[CHILDES_fre_table['Word'] == word]['Norm_freq_per_million'].item())

        except:
            # remove the corresponding row in the list
            freq_frame = freq_frame[freq_frame['word'] != word]

    freq_frame['Audiobook_freq_per_million'] = audiobook_lst
    freq_frame['CELEX_freq_per_million'] = CELEX_lst
    freq_frame['CHILDES_freq_per_million'] = CHILDES_lst
    freq_frame['Length'] = freq_frame['word'].apply(len)
    # get log 10
    def convert_to_log(freq):
        return math.log10(freq)

    freq_frame['CELEX_log_freq_per_million'] = freq_frame['CELEX_freq_per_million'].apply(convert_to_log)
    freq_frame['Audiobook_log_freq_per_million'] = freq_frame['Audiobook_freq_per_million'].apply(convert_to_log)
    freq_frame['CHILDES_log_freq_per_million'] = freq_frame['CHILDES_freq_per_million'].apply(convert_to_log)

    return freq_frame


def match_freq(CDI, audiobook, match_mode, num_bins, threshold):
    """
    match machine CDI with human CDI based on different match modes
        the range of the length
    """

    CDI, audiobook = match_range(CDI, audiobook)
    CDI_bins, matched_CDI = get_equal_bins(CDI['CHILDES_log_freq_per_million'].tolist(), CDI, num_bins)

    if match_mode == 'range_aligned':
        _, matched_audiobook = get_equal_bins(audiobook['Audiobook_log_freq_per_million'].tolist(), audiobook, num_bins)

    elif match_mode == 'bin_range_aligned':
        _, matched_audiobook = match_bin_range(CDI_bins, audiobook['Audiobook_log_freq_per_million'].tolist(),
                                               audiobook)

    elif match_mode == 'density_aligned':
        audiobook_bins_temp, matched_audiobook_temp = match_bin_range(CDI_bins, audiobook[
            'Audiobook_log_freq_per_million'].tolist(), audiobook)
        threshold_all = threshold * num_bins
        _, matched_audiobook = match_bin_density(matched_CDI, matched_audiobook_temp, CDI_bins, audiobook_bins_temp,
                                                 threshold_all)

    elif match_mode == 'distr_aligned':
        _, matched_audiobook_temp = match_bin_range(CDI_bins, audiobook['Audiobook_log_freq_per_million'].tolist(),
                                                    audiobook)
        matched_audiobook = match_bin_prop(matched_audiobook_temp, threshold)

    return matched_CDI, matched_audiobook


def compare_histogram(matched_CDI, matched_audiobook, num_bins, freq_type, lang, eval_condition, dataPath, match_mode,
                      machine_set, alpha=0.5):
    """
    plot figs and get freq stat of the matched groups
    """

    stat_audiobook = plot_density_hist(matched_audiobook, 'Audiobook_' + freq_type, freq_type, 'Machine', alpha,
                                       'given_bins', num_bins)
    stat_CHILDES = plot_density_hist(matched_CDI, 'CHILDES_' + freq_type, freq_type, 'Human', alpha, 'given_bins',
                                     num_bins)

    plt.legend()
    plt.title(lang + ' ' + eval_condition + ' (' + match_mode + ')')

    # save the plot to the target dir
    OutputPath = dataPath + '/' + match_mode + '/' + str(num_bins) + '_' + machine_set + '/Compare/'
    if not os.path.exists(OutputPath):
        os.makedirs(OutputPath)

    # concat results
    stat_audiobook['freq_type'] = 'Audiobook_' + freq_type
    stat_CHILDES['freq_type'] = 'CHILDES_' + freq_type
    stat_audiobook['set_type'] = 'Machine'
    stat_CHILDES['set_type'] = 'Human'
    stat_all = pd.concat([stat_audiobook, stat_CHILDES], ignore_index=True)

    plt.savefig(OutputPath + 'Compare_' + lang + '_' + eval_condition + '_' + freq_type, dpi=800)
    plt.clf()

    return stat_all


def plot_all_histogram(freq, num_bins, freq_type, lang, eval_condition, dataPath, match_mode, eval_type, machine_set,
                       alpha=0.5):
    """
    plot each freq stat with equal-sized bins
    """

    plot_density_hist(freq, 'Audiobook_' + freq_type, freq_type, 'Audiobook', alpha, 'equal_bins',
                      num_bins)
    plot_density_hist(freq, 'CHILDES_' + freq_type, freq_type, 'CHILDES', alpha, 'equal_bins', num_bins)
    plot_density_hist(freq, 'CELEX_' + freq_type, freq_type, 'CELEX', alpha, 'equal_bins', num_bins)
    plt.legend()
    plt.title(lang + ' ' + eval_condition + ' ' + eval_type + ' (' + match_mode + ')')

    # save the plot to the target dir
    OutputPath = dataPath + '/' + match_mode + '/' + str(num_bins) + '_' + machine_set + "/Stat/"

    if not os.path.exists(OutputPath):
        os.makedirs(OutputPath)

    plt.savefig(OutputPath + eval_type + '_' + lang + '_' + eval_condition + '_' + freq_type, dpi=800)
    plt.clf()


def main(argv):
    # Args parser
    args = parseArgs(argv)

    freq_path = args.outPath + 'matched_set/' + args.word_format + '/' + args.match_mode + '/' + str(
        args.num_bins) + '/'
    if not os.path.exists(freq_path):
        os.makedirs(freq_path)

    # step 1: load overall freq data
    CDI_freqOutput = freq_path + '/human_' + args.lang + '_' + args.eval_condition + '.csv'
    matched_freqOutput = freq_path + '/machine_' + args.lang + '_' + args.eval_condition + '.csv'

    # check whether there exists the matched data
    if os.path.exists(CDI_freqOutput) and os.path.exists(matched_freqOutput):
        print('There exist the matched CDI and audiobook freq words! Skip')
        matched_CDI: TextFileReader | DataFrame | Any = pd.read_csv(CDI_freqOutput)
        matched_audiobook = pd.read_csv(matched_freqOutput)

    # step 2: match the freq
    else:
        print('Creating the matched CDI and audiobook freq words')
        train_path = args.dataPath + 'freq_corpus/' + args.word_format + '/'
        testPath = args.dataPath + 'human_CDI/' + args.lang + '/' + args.eval_condition
        for file in os.listdir(testPath):
            CDI_test = pd.read_csv(testPath + '/' + file)

        CDI = get_freq_frame(CDI_test, train_path, args.word_type)

        audiobook_test = pd.read_csv(train_path + '/machine_' + args.machine_set + '.csv')

        audiobook = get_freq_frame(audiobook_test, train_path, args.word_type)

        matched_CDI, matched_audiobook = match_freq(CDI, audiobook, args.match_mode, args.num_bins, args.threshold)
        # save the filtered results
        matched_CDI.to_csv(CDI_freqOutput)
        matched_audiobook.to_csv(matched_freqOutput)

    # step 3: plot the distr figures
    fig_path = args.outPath + 'fig/' + args.word_format
    stat_path = args.outPath + '/stat/' + args.word_format + '/' + args.match_mode + '/' + str(
        args.num_bins) + '_' + args.machine_set + '/'
    # plot out the matched freq results
    if not os.path.exists(stat_path):
        os.makedirs(stat_path)

    stat_all = compare_histogram(matched_CDI, matched_audiobook, args.num_bins, args.freq_type, args.lang,
                                 args.eval_condition,
                                 fig_path, args.match_mode, args.machine_set, alpha=0.5)
    '''
    plot_all_histogram(matched_CDI, args.num_bins, args.freq_type, args.lang, args.eval_condition, fig_path,
                       args.match_mode, 'CDI', args.machine_set, alpha=0.5)
    plot_all_histogram(matched_audiobook, args.num_bins, args.freq_type, args.lang, args.eval_condition,
                       fig_path, args.match_mode, 'matched', args.machine_set, alpha=0.5)
    
    '''


    stat_all.to_csv(stat_path + args.lang + '_' + args.eval_condition + '.csv')

    print('Finished getting stat!')


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)