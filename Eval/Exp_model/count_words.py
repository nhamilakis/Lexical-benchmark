# -*- coding: utf-8 -*-
"""
explore generated tokens

@author: Jing Liu
"""

import pandas as pd
import os
import collections

def get_distr(root_path):
    
    '''
    get the entropy distr and KL divergence with the reference data for each type 
    
    input: the root directory containing all the generated files adn train reference
    output: word freq list with all the words and their counts
            dataframe with all the words across different conditions
    
    '''
    
    
    # load the rerference data
    
    month_lst = []
    prompt_lst = []
    h_all = []
    prob_all =[]
    strategy_lst = []
    beam_lst = []
    topk_lst = []
    temp_lst = []
    # go over the generated files recursively
    root_path = 'eval'
    
    
    for month in os.listdir(root_path): 
        if not month.endswith('.csv') and not month.endswith('.ods'): 
            train_distr = pd.read_csv(root_path + '/' + month + '/train_distr.csv')
            train_distr['month'] = month
            
            for prompt_type in os.listdir(root_path + '/' + month): 
                
                
                if not prompt_type.endswith('.csv') and not prompt_type.endswith('.ods'):                    
                    for strategy in os.listdir(root_path + '/' + month+ '/' + prompt_type): 
                        
                        for file in os.listdir(root_path + '/' + month+ '/' + prompt_type+ '/' + strategy):
                            try:      # in the case that entroyp and prob are not calculated yet
                                # load decoding strategy information
                                
                                data = pd.read_csv(root_path + '/' + month+ '/' + prompt_type + '/' + strategy + '/' + file)
                                result = []
                                n = 0
                                while n < data.shape[0]:
                                    generated = data['LSTM_segmented'].tolist()[n].split(' ')
                                    result.extend(generated)
                                    n += 1
                                
                                # get freq lists
                                frequencyDict = collections.Counter(result)  
                                freq_lst = list(frequencyDict.values())
                                word_lst = list(frequencyDict.keys())
                                
                                fre_table = pd.DataFrame([word_lst,freq_lst]).T
                                col_Names=["Word", "Freq"]
                                fre_table.columns = col_Names
                                
                                
                                
                                if strategy == 'beam':
                                    beam_lst.append(file.split('_')[0])
                                    topk_lst.append('0')
                                    
                                if strategy == 'top-k':
                                    topk_lst.append(file.split('_')[0])
                                    beam_lst.append('0')
                                
                                    # concatnete all the basic info regarding the genrated seq
                                strategy_lst.append(strategy)
                                prompt_lst.append(prompt_type)
                                month_lst.append(month)
                                temp_lst.append(file.split('_')[1])
                                
                            
                            except:
                                print(file)
                                
    info_frame = pd.DataFrame([month_lst,strategy_lst,beam_lst,topk_lst,temp_lst,h_all,prob_all,prompt_lst]).T
    
    # rename the columns
    info_frame.rename(columns = {0:'month', 1:'decoding', 2:'beam', 3:'top-k', 4:'temp',5:'entropy',6:'prob',7:'prompt'}, inplace = True)

    return info_frame, reference_frame


info_frame, reference_frame = get_distr('eval')