# -*- coding: utf-8 -*-
"""
Align the BabySLM words with CHILDES

Step1: recollect WORDANK

Step2: aggregate SLM baby(dev + test)

Step3: get model material
   1) freq
   2) POS annotation
   3) filter non-word pairs

Step4: get freq bins

Step5: plot results

Acumulator: phonmes freq
"""
import pandas as pd
import numpy as np
import statistics
import spacy


WG_s = pd.read_csv('CHILDES/AE_compre_WG-short.csv')
WG = pd.read_csv('log-figure/Selected_words_infant.csv')

# match freq range



# define function to compute vector of stats for a given bin
def compute_bin_stats(data):
    median = np.median(data)
    q1, q3 = np.percentile(data, [25, 75])
    min_val = np.min(data)
    max_val = np.max(data)
    #return [min_val, q1, median, q3, max_val]
    return median

'''
def chunk_list(lst, n):
    """Chunk a list of ordered numbers into n subgroups so that the median intervals of consecutive groups are as close as possible."""
    
    if n <= 0:
        raise ValueError('n must be a positive integer')
    if len(lst) < n:
        raise ValueError('list length must be at least n')
    
    chunk_size = len(lst) // n  # initial chunk size
    remainder = len(lst) % n  # remaining elements
    start_idx = 0
    chunks = []
    
    # iterate over n chunks
    for i in range(n):
        # calculate end index of chunk
        end_idx = start_idx + chunk_size
        # add remaining elements to first few chunks
        if remainder > 0:
            end_idx += 1
            remainder -= 1
        # create chunk and add to chunks list
        chunk = lst[start_idx:end_idx]
        chunks.append(chunk)
        # update start index for next chunk
        start_idx = end_idx
    
    
    # adjust chunks to minimize median intervals of consecutive groups
    while True:
        medians = [np.median(chunk) for chunk in chunks]  # calculate medians of each chunk
        intervals = [medians[i+1] - medians[i] for i in range(n-1)]  # calculate median intervals between consecutive chunks
        max_interval_idx = intervals.index(max(intervals))  # find index of chunk with largest median interval
        if max(intervals) <= 1:  # if maximum interval is already <= 1, break loop
            break
        # move element from chunk with largest median interval to neighboring chunk
        if len(chunks[max_interval_idx]) > 1:
            move_element = chunks[max_interval_idx].pop(0)
            chunks[max_interval_idx+1].append(move_element)
        else:
            move_element = chunks[max_interval_idx-1].pop(-1)
            chunks[max_interval_idx].insert(0, move_element)
    
    return chunks

'''

def chunk_list(lst,group_num):
    n = len(lst)
    chunk_size = int(np.ceil(n/group_num))

    groups = []
    for i in range(0, n, chunk_size):
        group = lst[i:i+chunk_size]
        groups.append(group)

    while len(groups) > group_num:
        new_groups = []
        for i in range(0, len(groups)-1, 2):
            merged_group = groups[i] + groups[i+1]
            new_group_size = int(np.ceil(len(merged_group)/2))
            new_groups.extend([merged_group[:new_group_size], merged_group[new_group_size:]])
        if len(groups) % 2 != 0:
            new_groups.append(groups[-1])
        groups = new_groups

    medians = [np.median(group) for group in groups]
    median_diffs = [abs(medians[i+1] - medians[i]) for i in range(len(medians)-1)]
    while len(groups) < group_num:
        max_diff_index = median_diffs.index(max(median_diffs))
        split_group = groups[max_diff_index]
        new_group_size = int(np.ceil(len(split_group)/2))
        groups[max_diff_index:max_diff_index+1] = [split_group[:new_group_size], split_group[new_group_size:]]
        medians = [np.median(group) for group in groups]
        median_diffs = [abs(medians[i+1] - medians[i]) for i in range(len(medians)-1)]

    return groups


def select_POS(selected_words, lang, word_type):
    
    if lang == 'EN': 
        nlp = spacy.load('en_core_web_sm')
    elif lang == 'FR': 
        nlp = spacy.load('fr_core_news_sm')    
        
    pos_all = []
    for word in selected_words['word']:     
        doc = nlp(word)
        pos_lst = []
        for token in doc:
            pos_lst.append(token.pos_)
        pos_all.append(pos_lst[0])
    selected_words['POS'] = pos_all
    
    func_POS = ['PRON','SCONJ','CONJ','CCONJ','DET','AUX', 'INTJ','PART']
    if word_type == 'all':
        selected_words = selected_words
    elif word_type == 'content':
        selected_words = selected_words[~selected_words['POS'].isin(func_POS)]
    elif word_type == 'func':
        selected_words = selected_words[selected_words['POS'].isin(func_POS)]
    return selected_words    



# inputs are dataframes 
def adjust_median(numbers,target_median):
    # Desired maximum absolute difference between current median and target median
    max_median_diff = 0.1 * target_median

    # Calculate the current median of the list
    current_median = numbers[len(numbers)//2]
    
    # Check if the current median is already close enough to the target median
    if abs(current_median - target_median) <= max_median_diff:
        print("List already has desired median")
    else:
        # Calculate the difference between the current median and the target median
        median_diff = current_median - target_median
    
        # Remove elements from the list until the median is close enough to the target median
        while abs(median_diff) > max_median_diff:
            if median_diff > 0:
                
                # Remove highest numbers that have greatest deviation from median
                numbers.pop(-1)
            else:
                # Remove lowest numbers that have greatest deviation from median
                numbers.pop(0)
    
            # Recalculate the median and difference
            current_median = numbers[len(numbers)//2]
            median_diff = current_median - target_median
    
        
    median = numbers[len(numbers)//2]
    return numbers,median   

def map_filename(df,candidate):
    m = 0
    final_frame = pd.DataFrame()   
    while m < df.shape[0]:
        selected_row = candidate[candidate['word'] == df['word'].tolist()[m]]
        #index_lst = selected_row['index'].tolist()
        index_lst = selected_row.index
        # fill in with the index
        indices = list(range(index_lst[0],index_lst[-1]+2))
        subframe = candidate.iloc[indices]
        
        final_frame = pd.concat([final_frame,subframe])
        m += 1
    return final_frame

lang = 'EN'
human_cdi = pd.read_csv('Selected_words_infant.csv')
#machine_cdi = pd.read_csv('Model_all.csv')
machine_cdi = pd.read_csv('Eval_test.csv')
machine_cdi = machine_cdi.sort_values('Log_norm_freq_per_million')
machine_groups = select_POS(machine_cdi, lang, 'content')


infant_prod = pd.read_csv('AE_production_freq_selected.csv')
freq_lst = human_cdi['Log_norm_freq_per_million'].tolist()
chunked_group = chunk_list(freq_lst,2)
'''
machine_groups.to_csv('Selected_words_model_clean.csv')   
candidate = pd.read_csv('test.csv')    
machine_groups= pd.read_csv('Selected_words_model_clean.csv')   
'''

'''
similar number of wordsin each freq bin
input:
output:
'''



def align_human_model(human_cdi,machine_groups,group_num,non_words):
    
    machine_groups = machine_groups[machine_groups['non word counts'] > non_words]
    # filter non_words
    groups = chunk_list(human_cdi['Log_norm_freq_per_million'].tolist(),group_num)
    median_lst_infant = []
    range_lst_infant = []
    for group in groups: 
        median = statistics.median(group)
        
        group_range = [min(group),max(group)]
        median_lst_infant.append(median)
        range_lst_infant.append(group_range) 
    
    # get the list of group name
    
    # assign group names
    machine_median_lst = []
    model_all = []
    n = 0
    while n < len(range_lst_infant):
        selected_frame = machine_groups[(machine_groups['Log_norm_freq_per_million'] >= range_lst_infant[n][0]) & (machine_groups['Log_norm_freq_per_million'] <= range_lst_infant[n][1])]
        machine_median =  statistics.median(selected_frame['Log_norm_freq_per_million'].tolist())
        # adjust medians
        numbers,median = adjust_median(selected_frame['Log_norm_freq_per_million'].tolist(),median_lst_infant[n])
        
        machine_median_lst.append(machine_median)
        df = pd.DataFrame()  
        
        for freq in numbers:
            selected_row = machine_groups[machine_groups['Log_norm_freq_per_million'] == freq]
            # update the grouped_model so that there is no duplicate lines
            first_row_subdf = selected_row.iloc[0:1]    
            # Remove the row where Column1 value is 2
            machine_groups = machine_groups[machine_groups['word'] != first_row_subdf['word'].tolist()[0]]
            #selected_row = selected_row.head(0)
            df = pd.concat([df,first_row_subdf])
        
        model_all.append(df)    
        n += 1
    return median_lst_infant,range_lst_infant,machine_median_lst, model_all, machine_median_lst

median_lst_infant,range_lst_infant,machine_median_lst, model_all, machine_median_lst = align_human_model(human_cdi,machine_groups,6, 4)

selected_words_model = pd.DataFrame()

n = 0
while n < len(model_all):
    
    model_all[n]['group'] = machine_median_lst[n]
    selected_words_model = pd.concat([selected_words_model,model_all[n]])
    n += 1
    
selected_words_model.to_csv('selected_words_model_filtered.csv')    
    
word_lst = [model_all[0]['Log_norm_freq_per_million'].tolist(),model_all[1]['Log_norm_freq_per_million'].tolist()]
final_freq_lst = [item for sublist in word_lst for item in sublist]   


    
# plot out frames in different freq bands
def align_material(model_all,candidate):
    m = 0
    for df in model_all:
        final_frame = pd.DataFrame()    
        n = 0
        while n < df.shape[0]:
            try:
                selected_row = candidate[candidate['word'] == df['word'].tolist()[n]]
                #index_lst = selected_row['index'].tolist()
                index_lst = selected_row.index
                # fill in with the index
                indices = list(range(index_lst[0],index_lst[-1]+2))
                subframe = candidate.iloc[indices]
                # remove the additional non-words
                id_list = subframe['id'].tolist()
                nonwords = list(set(id_list))[:6]
                # only take first 6 nonword id
                selected_frames = subframe[subframe['id'].isin(nonwords)]
                final_frame = pd.concat([final_frame,selected_frames])
            except:
                pass
            n += 1
        # remove duplicates
        final_frame.drop_duplicates(subset=['filename'], keep='first', inplace=True)
        final_frame.to_csv('material_selected' + str(m) + '.csv')
        m += 1




candidate = pd.read_csv('gold_no_filter.csv')
align_material(model_all,candidate)

def get_overlap(machine_groups):
    # find the reserved list
    n = 0
    reserve_lst = []
    while n < machine_groups.shape[0]:
        word_index = human_cdi[human_cdi['item_definition'] == machine_groups['word'].tolist()[n]]
        if word_index.shape[0] > 0:
            reserve_lst.append(1)
        else:
            reserve_lst.append(0)   
        n += 1     
    return reserve_lst
 



LM = pd.read_csv()
    
    
