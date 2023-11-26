# -*- coding: utf-8 -*-
"""
explore generated tokens

@author: Jing Liu
"""

import pandas as pd
import os
import collections
import enchant

d = enchant.Dict("en_US")

def match_seq(cleaned_word_temp,frame_all):
    '''
    get the entropy distr and KL divergence with the reference data for each type 
    
    input: the root directory containing all the generated files adn train reference
    output: dataframe with all the sequences across different conditions
            dataframe with all the words across different conditions
    '''
    
    cleaned_word_lst = ['MONTH','PROMPT','BEAM','TOPK']
    cleaned_word_lst.extend(cleaned_word_temp)
    # construct a total dataframe
    cleaned_frame = pd.DataFrame(cleaned_word_lst).T
    # Set the first row as column names
    cleaned_frame.columns = cleaned_frame.iloc[0]
    # Drop the first row
    cleaned_frame = cleaned_frame[1:].reset_index(drop=True)
    
    # Fill the DataFrame with zeros for the specified number of rows
    for i in range(len(frame_all)):
        cleaned_frame.loc[i] = [0] * len(cleaned_word_lst)
    
    i = 0
    while i < len(frame_all):
        # loop over the word freq frame 
        n = 0
        while n < frame_all[i].shape[0]:   
            
            word = frame_all[i]['Word'].tolist()[n]
            freq = frame_all[i]["Freq"].tolist()[n]       
            # append the list to the frame 
            cleaned_frame.loc[i,word] = freq
            n += 1
            
            
        # loop the parameter list
        for para in ['MONTH','PROMPT','BEAM','TOPK']:
            cleaned_frame.loc[i,para] = frame_all[i][para].tolist()[1]
        i += 1
        
    return cleaned_frame




def get_distr(root_path):
    
    '''
    get the entropy distr and KL divergence with the reference data for each type 
    
    input: the root directory containing all the generated files adn train reference
    output: dataframe with all the sequences across different conditions
            dataframe with all the words across different conditions
    '''
    frame_all = []
    seq_all = []
    # go over the generated files recursively
   
    for month in os.listdir(root_path): 
        if not month.endswith('.csv') and not month.endswith('.ods'): 
            
            for prompt_type in os.listdir(root_path + '/' + month): 
                
                if not prompt_type.endswith('.csv') and not prompt_type.endswith('.ods'):                    
                    for strategy in os.listdir(root_path + '/' + month+ '/' + prompt_type): 
                        
                        for file in os.listdir(root_path + '/' + month+ '/' + prompt_type+ '/' + strategy):
                            try:      
                                # load decoding strategy information
                                
                                data = pd.read_csv(root_path + '/' + month+ '/' + prompt_type + '/' + strategy + '/' + file)
                                seq = []
                                n = 0
                                while n < data.shape[0]:
                                    generated = data['LSTM_segmented'].tolist()[n].split(' ')
                                    seq.extend(generated)
                                    n += 1
                                
                                # get freq lists
                                frequencyDict = collections.Counter(seq)  
                                freq_lst = list(frequencyDict.values())
                                word_lst = list(frequencyDict.keys())
                                fre_table = pd.DataFrame([word_lst,freq_lst]).T
                                
                                col_Names=["Word", "Freq"]
                                
                                fre_table.columns = col_Names
                                
                                if strategy == 'beam':
                                    fre_table['BEAM'] = file.split('_')[0]
                                    fre_table['TOPK'] ='0'
                                    
                                elif strategy == 'top-k':
                                    fre_table['TOPK'] = file.split('_')[0]
                                    fre_table['BEAM'] ='0'
                                
                                fre_table['PROMPT'] = prompt_type
                                fre_table['MONTH'] = month
                                frame_all.append(fre_table)
                                seq_all.extend(seq)
                                
                            except:
                                print(file)
                                
    seq_lst = list(set(seq_all))
    
    seq_frame = match_seq(seq_lst,frame_all)
    
    # select the word from the dataframe
    word_lst = []
    for seq in seq_lst:
        try: 
            if d.check(seq) == True:
                word_lst.append(seq)
        except:
            print(seq)
    word_lst.extend(['MONTH','PROMPT','BEAM','TOPK'])
    word_frame = seq_frame[word_lst]
    return seq_frame, word_frame


seq_frame, word_frame = get_distr('eval_1')


# plot the figures 
df = df.drop(columns=['MONTH','PROMPT','BEAM','TOPK'])
threshold = 