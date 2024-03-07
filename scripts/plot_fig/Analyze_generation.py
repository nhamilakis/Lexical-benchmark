#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The oov tokens metric

@author: jliu
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import enchant

d = enchant.Dict("en_US")

sns.set_style('whitegrid')

def get_fre_table(file_path):
    
    with open(file_path, encoding="utf8") as f:
        sent_lst = f.readlines()
    word_lst = []    
    for sent in sent_lst:
        # remove the beginning and ending space
        words = sent.split(' ')
        for word in words:
            cleaned_word = word.strip()
            if len(cleaned_word) > 0:  
                word_lst.append(cleaned_word)
    return word_lst

def plot_temp(root_path, prompt_lst,input_type):
    for prompt in prompt_lst:
        for temp in os.listdir(root_path + prompt):
            data = pd.read_csv(root_path + prompt + '/' + str(temp) +'/prop.csv')
            data = data.sort_values(by='model')
            sns.lineplot(data['model']/89,data['oov_' + input_type + '_prop'], linewidth=3.5
                         , label= prompt + '_' + str(temp))
    plt.legend()
    if input_type == 'token':
        plt.ylim(0,1)
    plt.xlabel('(Pseudo) age in month', fontsize=15)
    plt.ylabel('oov token prop', fontsize=15)
    plt.title('Temperature effect on OOV ' + input_type, fontsize=15, fontweight='bold') 

root_path = '/data/exp/oov_words/' 
prompt_lst = ['prompted','unprompted']
input_type = 'word'
plot_temp(root_path, prompt_lst,input_type)



def plot_CHILDES(prop_frame_path,input_type):
    prop_frame = pd.read_csv(prop_frame_path)
    month_lst = [int(x) for x in prop_frame['month'].tolist()]
    prop_lst = [float(x) for x in prop_frame['oov_' + input_type].tolist()]
    sns.lineplot(month_lst,prop_lst, linewidth=3.5)
    if input_type == 'token':
        plt.ylim(0,1)
    else:
        plt.ylim(0,0.2)
    plt.xlabel('Age in month', fontsize=15)
    plt.ylabel('oov ' + input_type +' prop', fontsize=15)
    plt.title('OOV '+ input_type +' in CHILDES', fontsize=15, fontweight='bold') 
    
prop_frame_path = '/data/exp/oov_words/largest_set/oov_CHILDES/prop.csv'
plot_CHILDES(prop_frame_path,'word')

'''
analyze training set in different chunks

get the avg freq 

get #nonwords or prop
'''
lang_lst = ['AE','BE']
hour_lst = ['400','800','1600','3200']
lang = 'AE'
machine_CDI_path = '/data/Machine_CDI/Lexical-benchmark_output/test_set/matched_set/char/bin_range_aligned/6_audiobook_aligned/machine_'+ lang + '_exp.csv'
freq_path = '/data/Machine_CDI/Lexical-benchmark_data/exp/Audiobook/freq_by_train/'
machine_CDI = pd.read_csv(machine_CDI_path)
freq_frame = pd.read_csv(freq_path + hour_lst[-1] + '.csv')
machine_CDI_grouped = machine_CDI.groupby('group')

for group,machine_CDI_group in machine_CDI_grouped:
    group_freq = []
    # get the selected words in the freq bands
    for hour in hour_lst:
        freq_frame = pd.read_csv(freq_path + hour + '.csv')
        # get intersected words; remove nonwords for the time being
        #inter_words = 
        
        
'''
analyze the generated tokens
'''

train_path = '/data/Lexical-benchmark_backup/Output/CDI/AE/recep/'
group_stat_path = '/data/Machine_CDI/Lexical-benchmark_output/CHILDES/CDI/AE/exp/stat_per_file.csv'
OutputPath = '/data/Machine_CDI/Lexical-benchmark_output/CHILDES/CDI/AE/exp'
month_corre = True
analysis_type = 'number'

freqpath_root = '/data/exp/oov_words/' + analysis_type + '/'

def count_oov_tokens(freqpath_root,train_path,group_stat_path,OutputPath,month_corre,analysis_type):
    
    '''
    count words of CHILDES
    
    input: freq table of exposure and generation contents
    return: the oov token prop and oov type prop
    '''
    if not month_corre:
        train = pd.read_csv(train_path+'all_freq.csv', index_col=0)
        folder = 'largest_set'
        word_lst = train.index
        
    group_stat = pd.read_csv(group_stat_path, index_col=0)
    prop_frame = pd.DataFrame()
    # loop each month
    for file in set(group_stat['month']):
        
        # get word freq list for each file
        text_file = 'transcript_' + str(file)
        file_path = OutputPath + '/Transcript_by_month/' + text_file + '.txt'  
        if month_corre:
            train_text_path = train_path + '/Transcript_by_month/' + text_file + '.txt'  
            word_lst = get_fre_table(train_text_path)
            folder = 'month_correspondent'
            # output a dataframe with the word freq
            
        with open(file_path, encoding="utf8") as f:
            sent_lst = f.readlines()
        
        if analysis_type == 'type':
            seq_month = set()
            oov_tokens = set()
            oov_words = set()
            for sent in sent_lst:
                # remove the beginning and ending space
                words = sent.split(' ')
                generated = [x for x in words if x != ""]
                for word in generated:
                    if not word.isspace():
                        seq_month.add(word)
                        if word not in word_lst:
                            oov_tokens.add(word)
                            if d.check(word) == True:
                                oov_words.add(word)
                                
        else:
            seq_month = []
            oov_tokens = []
            oov_words = []
            for sent in sent_lst:
                # remove the beginning and ending space
                words = sent.split(' ')
                generated = [x for x in words if x != ""]
                for word in generated:
                    if not word.isspace():
                        seq_month.append(word)
                        if word not in word_lst:
                            oov_tokens.append(word)
                            if d.check(word) == True:
                                oov_words.append(word)
            
        print('Finish month ' + str(file))
        # get the prop of the oov words
        if len(seq_month) != 0:
            prop_frame_single = pd.DataFrame([str(file),len(oov_words)/len(seq_month),
                                          1 - len(oov_words)/len(seq_month),len(oov_tokens)/len(seq_month)])
            prop_frame = pd.concat([prop_frame,prop_frame_single.T])
            
        else:
            print(str(file))
        freqpath = freqpath_root + folder + '/oov_CHILDES/'
        
        
        if not os.path.exists(freqpath):
            os.makedirs(freqpath)

        
        seq_frame_month = pd.DataFrame(seq_month)
        seq_frame_month['month'] = str(file)
        seq_frame_month.to_csv(freqpath + 'seq_' + str(file) + '.csv')
        oov_token_frame = pd.DataFrame(oov_tokens)
        oov_token_frame['month'] = str(file)
        oov_token_frame.to_csv(freqpath + 'token_' + str(file)+ '.csv')
        oov_word_frame = pd.DataFrame(oov_words)
        oov_word_frame['month'] = str(file)
        oov_word_frame.to_csv(freqpath + 'word_' + str(file) + '.csv')
     
    prop_frame.columns = ['month','oov_word_prop','oov_nonword_prop','oov_token_prop']
    prop_frame.to_csv(freqpath + 'prop.csv')  
    
    return freq_frame
       
        
       
        


        