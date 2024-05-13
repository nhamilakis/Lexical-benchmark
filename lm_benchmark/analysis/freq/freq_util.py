#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import collections
import random
from itertools import islice
from collections import deque
import statistics

random.seed(45)


def get_freq_table(word_lst):
    """get freq from a word list"""
    # clean text and get the freq table
    result = [str(word) for sentence in word_lst for word in str(sentence).split()]
    frequencyDict = collections.Counter(result)
    freq_lst = list(frequencyDict.values())
    word_lst = list(frequencyDict.keys())

    # get freq
    fre_table = pd.DataFrame([word_lst, freq_lst]).T
    col_Names = ["word", "count"]
    fre_table.columns = col_Names
    fre_table['freq_m'] = fre_table['count'] / len(result) * 1000000
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




def get_bin_stat(df, column_header:str):
    '''Get statistics of each bin'''
    # Group by 'group' column and calculate statistics
    stats_df = df.groupby('group').agg(
        count=('group', 'size'),
        min_value=(column_header, 'min'),
        max_value=(column_header, 'max'),
        mean_value=(column_header, 'mean'),
        median_value=(column_header, 'median'),
    ).reset_index()

    # Rename columns for clarity
    stats_df.rename(columns={'min_value': 'min', 'max_value': 'max', 'mean_value': 'mean', 'median_value': 'median'},
                    inplace=True)
    return stats_df

def get_equal_bins(data, data_frame, n_bins):
    '''
    get equal-sized bins
    input: a sorted array or a list of numbers;
        computes a split of the data into n_bins bins of approximately the same size
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
            CDI_selected, audiobook_selected = get_intersections(CDI_group, audiobook_group,
                                                                 'freq_m', 'freq_m')
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
    for i in range(0, len(bins) - 1):
        bin_membership[(audiobook >= bins[i]) & (audiobook <= bins[i + 1])] = i
    audiobook_frame['group'] = bin_membership
    CDI, audiobook_frame = align_group(CDI, audiobook_frame)

    if not match_median:
        return CDI, audiobook_frame

    else:
        # do the greedy search within each band
        pass

    # TODO: return the stat after matching


'''
# %%
import math
import pandas as pd

machine = pd.read_csv("/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/test/corpus/AE_machine.csv")[
    ["word", "freq_m"]
]

human = pd.read_csv("/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/test/corpus/AE_human.csv")[
    ["word", "freq_m"]
]
human.rename(columns={"freq_m": "logfreq"}, inplace=True)
machine.rename(columns={"freq_m": "logfreq"}, inplace=True)
# %%
band = [0.30, 28.51]
machine = machine[machine["logfreq"].between(*band)]
human = human[human["logfreq"].between(*band)]

# %%
hmean, hmedian, hmin, hmax = human["logfreq"].describe()[["mean", "50%", "min", "max"]]
c1, c2, c3, c4, c5 = 0.1, 0.1, 0.1, 0.1, 0.1

def cost(logfreqs):
    mmean, mmedian, mmin, mmax, count = pd.Series(logfreqs).describe()[
        ["mean", "50%", "min", "max", "count"]
    ]
    return (
        c1 * (hmean - mmean) ** 2
        + c2 * (hmedian - mmedian) ** 2
        + c3 * (hmin - mmin) ** 2
        + c4 * (hmax - mmax) ** 2
        - c5 * count / len(machine)
    )


logfreqs = []
prev_cost = math.inf
for _, row in machine.iterrows():
    current_cost = cost(logfreqs + [row["logfreq"]])
    if current_cost < prev_cost:
        logfreqs.append(row["logfreq"])
        prev_cost = current_cost

print("HUMAN")
print(pd.Series(human["logfreq"]).describe())
print()
print("MACHINE")
print(pd.Series(logfreqs).describe())
'''
