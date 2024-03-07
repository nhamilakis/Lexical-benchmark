#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
match generations with CHILDES data

@author: jliu
"""
import os 
import pandas as pd


def match_generation(generation,selected_frame):
    
    def count_token(x):
        return len(x.split(' '))-1   
    
    token_num_lst = []
    generation['num_tokens'] = generation['LSTM_segmented'].apply(count_token)
    unmatched_frame = pd.DataFrame()
    matched_frame = pd.DataFrame()
    rest_frame = pd.DataFrame()
    generation_grouped = generation.groupby('num_tokens')
    for token_num, generation_group in generation_grouped:
        # select first n rows
        candi_frame = selected_frame[selected_frame['num_tokens'] == token_num]
        
        if candi_frame.shape[0] < generation_group.shape[0]:
            token_num_lst.append(token_num)
            # collect rest of the rows
            rest_rows = generation_group[min(candi_frame.shape[0],generation_group.shape[0]):]
            rest_frame = pd.concat([rest_frame,rest_rows])
            
        matched_rows = candi_frame.head(min(candi_frame.shape[0],generation_group.shape[0]))
        # append the generated tokens by selecting the smallest number of rows
        matched_rows[prompt_type + '_' + temp] = generation_group.head(min(candi_frame.shape[0],generation_group.shape[0]))['LSTM_segmented'].tolist()
        matched_frame = pd.concat([matched_frame,matched_rows])
        unmatched_frame = pd.concat([unmatched_frame,candi_frame[min(candi_frame.shape[0],generation_group.shape[0]):]])
    
   
    # concatenate the rest of matched frame
    return matched_frame, unmatched_frame,token_num_lst



root_dir = '/data/exp/'
prompt_type = 'unprompted'
temp = '1.0'
root_path = root_dir + prompt_type + '_' + temp + '/'
# further select the age range of children's production
age_range = [6,36]
script = trans_speaker[(trans_speaker['month'] >= age_range[0]) & (trans_speaker['month'] <= age_range[1])]

# match the generations with human data
month_dict = {'50':[1],'100':[1],'200':[2,3],'400':[4,8],'800':[9,18],'1600':[19,28],'3200':[29,36]}

matched_all = pd.DataFrame()
unmatched_all = pd.DataFrame()
token_num_all = {}
for file in os.listdir(root_path):
    # select the subdataframe
    age_range = month_dict[file.split('.')[0]]
    selected_frame = script[(script['month'] >= age_range[0]) & (script['month'] <= age_range[1])]
    # load dataframe
    generation = pd.read_csv(root_path + file)
    matched_frame, unmatched_frame,token_num_lst = match_generation(generation,selected_frame)
    matched_all = pd.concat([matched_all,matched_frame])
    unmatched_all = pd.concat([unmatched_all,unmatched_frame])
    # concat log files
    if file in token_num_all:
        # If the key exists, append the new values to the existing list
        token_num_all[file].extend(token_num_lst)
    else:
        # If the key doesn't exist, create a new key-value pair with the key and values
        token_num_all[file] = token_num_lst
    

matched_all.to_csv('matched.csv')
unmatched_all.to_csv('unmatched.csv')


'''
add the larger model's generations
'''

script_all = pd.DataFrame()
root_dir = '/data/exp/'

generated = []
condi = 'unprompted_0.6/' 
for file in os.listdir(root_dir + condi):
    gen = pd.read_csv(root_dir + condi + file)
    script_all = pd.concat([script_all,gen])



script_all = script_all.rename(columns={'LSTM_segmented': condi})
script_all.pop('LSTM_generated')
# loop over the temp_lst
temp_lst = [ '1.0', '1.5']
for temp in temp_lst:
    generated = []
    condi = 'unprompted_' + temp
    for file in os.listdir(root_dir + condi + '/'):
        gen = pd.read_csv(root_dir + condi+ '/' + file)
        generated.extend(gen['LSTM_segmented'])  
    
    script_all[condi] = generated

generation = pd.read_csv('/data/exp/generation.csv')

# re-name files
script_all.pop('LSTM_generated')
script_all = script_all.rename(columns={'LSTM_segmented': 'unprompted_0.6'})






root_dir = '/data/exp/generation/4500h/00/unprompted/sample_random/'
for file in os.listdir(root_dir):
    script_new = pd.read_csv(root_dir + file)
    # replace columns in the dataframe
    script['unprompted_' + file.split('_')[1]] = script_new['LSTM_segmented']

script.pop('LSTM_segmented')
script.pop('LSTM_generated')



def order_df(script):

    # match different 
    start_index = script.columns.get_loc('speaker')
    
    # Select sub-dataframe from the starting column to the end
    sub_script= script.iloc[:, start_index:]
    
    return sub_script

script_ordered1 = order_df(script)
script_all_ordered = order_df(script_all)

script_final = pd.concat([script_all_ordered,script_ordered])
script_final = pd.concat([script_final,script_ordered1])
# Remove the last column
script_final = script_final.iloc[:, :-1]

# List of column headers to order
ordered_columns = ['unprompted_0.3', 'unprompted_0.6', 'unprompted_1.0', 'unprompted_1.5']

# Get the last n columns based on the length of the ordered_columns list
n = 4
last_n_columns = script_final.iloc[:, -n:]

# Reorder the last n columns based on the ordered_columns list
ordered_last_n_columns = last_n_columns[ordered_columns]

# Combine the ordered last n columns with the rest of the DataFrame
df_reordered = pd.concat([script_final.iloc[:, :-n], ordered_last_n_columns], axis=1)


df_reordered.to_csv('/data/exp/generation.csv')























