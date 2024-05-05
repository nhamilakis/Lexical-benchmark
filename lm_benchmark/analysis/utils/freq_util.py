#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
util func for get_stat

@author: jliu
"""

import pandas as pd
import math
import numpy as np
import collections
import matplotlib.pyplot as plt
import random
from itertools import islice
from collections import deque
import statistics

random.seed(45)


def get_freq_table(result):
    """get freq from a word list"""
    # clean text and get the freq table
    frequencyDict = collections.Counter(result)
    freq_lst = list(frequencyDict.values())
    word_lst = list(frequencyDict.keys())

    # get freq
    fre_table = pd.DataFrame([word_lst, freq_lst]).T
    col_Names = ["word", "count"]
    fre_table.columns = col_Names
    fre_table['freq'] = fre_table['count'] / len(result)
    return fre_table




def get_intersections(df1, df2, column1, column2):
    # align dataframe 1 with dataframe 2
    max_freq = min(df1[column1].max(), df2[column2].max())
    min_freq = max(df1[column1].min(), df2[column2].min())
    matched_df1 = df1[(df1[column1] >= min_freq) & (df1[column1] <= max_freq)]
    matched_df2 = df2[(df2[column2] >= min_freq) & (df2[column2] <= max_freq)]

    return matched_df1, matched_df2


def match_medians(word_freq_dict, target_median, target_len):
    '''
    set the tolerance threhsold on the results;
    return a selected nested list candidate
    '''

    def find_closest_number(lst, target):
        closest_number = lst[0]
        min_difference = abs(target - lst[0])

        for number in lst:
            difference = abs(target - number)
            if difference < min_difference:
                min_difference = difference
                closest_number = number

        return closest_number

    def divide_dict(candi_dict, target_len, condi):

        '''
        divide a dictionary based on target medians
        input: unsorted dictionary
        return two dictionaries
        '''
        if condi == 'freq':
            sorted_candi_dict = dict(sorted(candi_dict.items(), key=lambda item: item[1][0]))
            # get the closest num to target word
            rest_len_sorted = [length for length, _ in sorted_candi_dict.values()]

        else:
            sorted_candi_dict = dict(sorted(candi_dict.items(), key=lambda item: item[1][1]))
            # get the closest num to target word
            rest_len_sorted = [length for _, length in sorted_candi_dict.values()]

        # get the closest len num; assume it's in the rest of the list
        closest_number = find_closest_number(rest_len_sorted, target_len)
        # initialize left and right dict in the fixed dict
        index = rest_len_sorted.index(closest_number)

        left_dict = dict(islice(sorted_candi_dict.items(), index))
        right_dict = dict(deque(sorted_candi_dict.items(), maxlen=len(rest_len_sorted) - index))

        return left_dict, right_dict, index

    def generate_index(candi_dict, rest_dict, target_len, num):

        '''
        input: rest of index, index lst to select from, num, range_index
        return a list of selected index
        '''

        def select_index(candi_dict, index):

            def shuffle_dict(my_dict):
                # Get a list of key-value tuples and shuffle it
                items = list(my_dict.items())
                random.shuffle(items)
                # Convert the shuffled list back to a dictionary
                shuffled_dict = dict(items)
                return shuffled_dict

            candi_dict = shuffle_dict(candi_dict)

            selected_dict = dict(islice(candi_dict.items(), index))
            return selected_dict

        # divide the fixed rest dictionary into 2 parts
        left_dict_rest, right_dict_rest, _ = divide_dict(rest_dict, target_len, 'len')
        left_dict_candi, right_dict_candi, _ = divide_dict(candi_dict, target_len, 'len')
        # select the results based on fixed num
        added_num = abs(len(left_dict_rest) - len(right_dict_rest))
        # equally allocate the rest of index
        mod = (num - added_num) % 2
        allo_num = (num - added_num) // 2

        if len(left_dict_rest) >= len(right_dict_rest):
            # select the index to align right side
            right_candi_num = added_num + allo_num + mod
            left_candi_num = added_num

        elif len(right_dict_rest) > len(left_dict_rest):
            # select the index from left side
            left_candi_num = added_num + allo_num + mod
            right_candi_num = added_num

        # select from candi dict based on the number of index
        selected_left = select_index(left_dict_candi, left_candi_num)
        selected_right = select_index(right_dict_candi, right_candi_num)
        rest_dict.update(selected_left)
        rest_dict.update(selected_right)
        return rest_dict

    def get_range_dict(word_freq_dict):

        items = list(word_freq_dict.items())
        range_dict = {}
        len_lst = [length for _, length in word_freq_dict.values()]
        freq_lst = [freq for freq, _ in word_freq_dict.values()]

        # get range of freq and len lists
        min_len_index = len_lst.index(min(len_lst))
        max_len_index = len_lst.index(max(len_lst))
        min_freq_index = freq_lst.index(min(freq_lst))
        max_freq_index = freq_lst.index(max(freq_lst))

        # get key-value pairs
        range_dict[items[min_len_index][0]] = items[min_len_index][1]
        range_dict[items[max_len_index][0]] = items[max_len_index][1]
        range_dict[items[min_freq_index][0]] = items[min_freq_index][1]
        range_dict[items[max_freq_index][0]] = items[max_freq_index][1]

        return range_dict

    # check the threshold
    median_thre = target_median * 0.01
    # initialize the dict sorted by freq
    word_freq_dict = dict(sorted(word_freq_dict.items(), key=lambda item: item[1][0]))
    current_median = statistics.median_high([length for length, _ in word_freq_dict.values()])
    if not abs(current_median - target_median) > median_thre:
        return word_freq_dict

    else:

        left_dict, right_dict, index = divide_dict(word_freq_dict, target_median, 'freq')
        range_dict = get_range_dict(word_freq_dict)

        # update left or right dict
        if index > len(word_freq_dict) - index:
            print('remove indices in the left')
            # remove range datapoints from candi dict
            for key in range_dict:
                left_dict.pop(key, None)
            right_dict.update(range_dict)
            updated_dict = generate_index(left_dict, right_dict, target_len, len(word_freq_dict) - index)

        else:
            print('remove indices in the right')
            # put range index into the fixed dict
            for key in range_dict:
                right_dict.pop(key, None)
            left_dict.update(range_dict)
            updated_dict = generate_index(right_dict, left_dict, target_len, index)

        return updated_dict


def match_range(CDI, audiobook):
    '''
    match the audiobook sets with CHILDES of differetn modes
    Returns shrinked dataset with the matched range
    '''
    matched_CDI, matched_audiobook = get_intersections(CDI, audiobook, 'CHILDES_log_freq_per_million',
                                                       'Audiobook_log_freq_per_million')

    # sort the results by freq
    matched_CDI = matched_CDI.sort_values(by='CHILDES_log_freq_per_million')
    matched_audiobook = matched_audiobook.sort_values(by='Audiobook_log_freq_per_million')

    return matched_CDI, matched_audiobook


def get_bin_stat(bins, data_sorted):
    '''
    get stat of freq bins
    input: column with annotated group name
    return bin_stat
    '''
    data_sorted = np.array(data_sorted)
    # computing statistics over the bins (size, min, max, mean, med, low and high boundaries and density)
    boundaries = list(zip(bins[:-1], bins[1:]))
    binned_data_count = [len(data_sorted[(data_sorted >= l) & (data_sorted < h)]) for l, h in boundaries]
    binned_data_min = [np.min(data_sorted[(data_sorted >= l) & (data_sorted < h)]) for l, h in boundaries]
    binned_data_max = [np.max(data_sorted[(data_sorted >= l) & (data_sorted < h)]) for l, h in boundaries]
    binned_data_mean = [np.mean(data_sorted[(data_sorted >= l) & (data_sorted < h)]) for l, h in boundaries]
    binned_data_median = [np.median(data_sorted[(data_sorted >= l) & (data_sorted < h)]) for l, h in boundaries]
    bins_stats = pd.DataFrame(
        {'count': binned_data_count, 'min': binned_data_min, 'max': binned_data_max, 'mean': binned_data_mean,
         'median': binned_data_median})
    bins_stats['low'] = bins[:-1]
    bins_stats['high'] = bins[1:]
    bins_stats['density'] = bins_stats['count'] / (bins_stats['high'] - bins_stats['low']) / sum(bins_stats['count'])
    # Rename the newly created column to 'group'
    bins_stats = bins_stats.reset_index()
    bins_stats = bins_stats.rename(columns={'index': 'group'})

    return bins_stats


def get_len_stat(df, column_header):
    '''
    get stat of word length
    input: dataframe with annotated group name
    return bin_stat
    '''
    # Group by 'Group' column and calculate statistics
    stats_df = df.groupby('group')[column_header].agg(
        min='min',
        max='max',
        mean='mean',
        median='median'
    ).reset_index()

    stats_df.rename(columns={'min': 'len_min', 'max': 'len_max', 'mean': 'len_mean', 'median': 'len_median'},
                    inplace=True)
    return stats_df


def get_equal_bins(data, data_frame, n_bins):
    '''
    get equal-sized bins
    input: a sorted array or a list of numbers; computes a split of the data into n_bins bins of approximately the same size

    return
        bins: array with each bin boundary
        bins_stats
    '''
    # preparing data (adding small jitter to remove ties)
    size = len(data)
    assert n_bins <= size, "too many bins compared to data size"
    mindif = np.min(np.abs(np.diff(np.sort(np.unique(data)))))  # minimum difference between consecutive distinct values
    jitter = mindif * 0.01  # this small jitter will not change the relative order between datapoints
    data_jitter = np.array(data) + np.random.uniform(low=-jitter, high=jitter, size=size)
    data_sorted = np.sort(data_jitter)  # little jitter to remove ties

    # Creating the bins with approx equal number of observations
    bin_indices = np.linspace(1, len(data), n_bins + 1) - 1  # indices to edges in sorted data
    bins = [data_sorted[0]]  # left edge inclusive
    bins = np.append(bins, [(data_sorted[int(b)] + data_sorted[int(b + 1)]) / 2 for b in bin_indices[1:-1]])
    bins = np.append(bins, data_sorted[-1] + jitter)  # this is because the extreme right edge is inclusive in plt.hits

    # computing bin membership for the original data; append bin membership to stat
    bin_membership = np.zeros(size, dtype=int)
    for i in range(0, len(bins) - 1):
        bin_membership[(data_jitter >= bins[i]) & (data_jitter < bins[i + 1])] = i

    data_frame['group'] = bin_membership

    return bins, data_frame


def match_bin_range(CDI_bins, CDI, audiobook, audiobook_frame, match_median):
    '''
    match range of the audiobook freq of machine CDI with CHILDES freq of CDI

    input:
        human CDI eauql-sized bins
        machine-CDI freq bin
    Returns
        bins: machiine CDI with adjusted group array
        bins_stats: machine CDI dataframe with annotated group
    '''

    def align_group(CDI, audiobook_frame):

        matched_CDI = pd.DataFrame()
        matched_audiobook = pd.DataFrame()

        for group in set(audiobook_frame['group']):
            CDI_group = CDI[CDI['group'] == group]
            audiobook_group = audiobook_frame[audiobook_frame['group'] == group]
            CDI_selected, audiobook_selected = get_intersections(CDI_group, audiobook_group, 'word_len', 'word_len')
            matched_CDI = pd.concat([matched_CDI, CDI_selected])
            matched_audiobook = pd.concat([matched_audiobook, audiobook_selected])

        return matched_CDI, matched_audiobook

    def find_closest_numbers(arr, target_array):
        closest_numbers = [min(arr, key=lambda x: abs(x - target)) for target in target_array]
        return np.array(closest_numbers)

    audiobook = np.array(audiobook)
    # Creating the bins with approx equal number of observations
    bins = find_closest_numbers(audiobook, CDI_bins)
    # computing bin membership for the original data; append bin membership to stat
    bin_membership = np.zeros(len(audiobook), dtype=int)
    # replace the group name into target median
    '''
    for i in range(0,len(bins)-1):
       bin_membership[(audiobook>=bins[i])&(audiobook<=bins[i+1])]=i
    '''
    for i in range(0, len(bins) - 1):
        bin_membership[(audiobook >= bins[i]) & (audiobook <= bins[i + 1])] = i

    audiobook_frame['group'] = bin_membership
    CDI, audiobook_frame = align_group(CDI, audiobook_frame)

    if not match_median:
        return CDI, audiobook_frame

    else:
        # match freq and len medians of each freq bin
        target_frame_grouped = CDI.groupby('group')
        matched_audiobook = pd.DataFrame()

        for group, target_frame_group in target_frame_grouped:

            target_freq = target_frame_group['CHILDES_log_freq_per_million'].median()

            target_len = target_frame_group['word_len'].median()

            machine_group_frame = audiobook_frame[audiobook_frame['group'] == group]
            machine_group = {}
            for _, row in machine_group_frame.iterrows():
                key = row['word']
                values = (row['Audiobook_log_freq_per_million'], row['word_len'])
                machine_group[key] = values

            updated_dict = match_medians(machine_group, target_freq, target_len)

            frequencyDict = collections.Counter(updated_dict.values())
            count_lst = list(frequencyDict.values())
            freq_lst = list(frequencyDict.keys())
            # Randomize the row orders
            randomized_df = machine_group_frame.sample(frac=1)
            word_frame = pd.DataFrame()
            n = 0
            while n < len(freq_lst):
                selected_frame_words = randomized_df[randomized_df['Audiobook_log_freq_per_million'] == freq_lst[n][0]]
                selected_frame_words = selected_frame_words[selected_frame_words['word_len'] == freq_lst[n][1]]
                # generate the index randomly
                selected_frame_words = selected_frame_words.reindex()
                selected_frame = selected_frame_words.iloc[:count_lst[n]]
                word_frame = pd.concat([word_frame, selected_frame])
                n += 1

            matched_audiobook = pd.concat([matched_audiobook, word_frame])

        return CDI, matched_audiobook



