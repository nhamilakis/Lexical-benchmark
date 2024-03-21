#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oov analysis
use enchant_en as golden reference
@author: jliu
"""
import os
import pandas as pd
import enchant
import seaborn as sns
import matplotlib.pyplot as plt
d = enchant.Dict("en_US")
sns.set_style('whitegrid')



def generalize(exp_count_path,prod_count_path,output_path,prod_type):
    
    '''
    generalization analysis by comparing input and output tokens
    
    input:
        - generation word count csv
        - exposure csv
        
    return 
        - oov tokens/words by months: token \ count \ month \ word(Boolean)
        - inv tokens/words by months: token \ count \ month \ word(Boolean)
        - stat frame with each month data
        
    '''
    
    def get_intersect(true_exp, true_prod,prod_type):
        
        '''
        get the intersected and oov tokens and their counts
        return: 2 dataframe with 4 columns: word, count, month, word(Boolean)
        '''
        true_exp.name = 'count'
        true_prod.columns.values[0] = 'count'
        # Overlapping elements
        overlapping_elements = set(true_exp.index).intersection(set(true_prod.index))
        # Elements in list A but not in list B
        difference_elements = set(true_prod.index).difference(set(true_exp.index))
        # select the subdataframe based on index
        inv = true_prod.loc[overlapping_elements]
        oov = true_prod.loc[difference_elements]
        inv['prod_type'] = prod_type
        oov['prod_type'] = prod_type
        # get the stat frame
        oov_token = oov['count'].sum()/true_prod['count'].sum()
        oov_word = oov[oov['word']==True]['count'].sum()/true_prod['count'].sum()
        oov_token_type = oov.shape[0]/true_prod.shape[0]
        oov_word_type = oov[oov['word']==True].shape[0]/true_prod.shape[0]
        inv_word = inv[inv['word']==True]['count'].sum()/true_prod['count'].sum()
        inv_token_type = inv.shape[0]/true_prod.shape[0]
        inv_word_type = inv[inv['word']==True].shape[0]/true_prod.shape[0]
        stat = pd.DataFrame([prod_type,int(month),oov_token,oov_word,oov_token_type,oov_word_type,
                             1-oov_token,inv_word,inv_token_type,inv_word_type]).T
        stat.columns = ['prod_type','month','oov_token','oov_word','oov_token_type','oov_word_type',
                             'inv_token','inv_word','inv_token_type','inv_word_type']
        return inv, oov, stat
    
    # check whether the generated tokens are true words
    exp_count = pd.read_csv(exp_count_path, index_col=0)
    prod_count = pd.read_csv(prod_count_path, index_col=0)
    # get the intersected columns of df
    prod_count = prod_count.reset_index().dropna(subset=['index']).set_index('index')
    selected_months = set(exp_count.columns).intersection(set(prod_count.columns))
    exp_count = exp_count[list(selected_months)]
    prod_count = prod_count[list(selected_months)]
    word_lst = []
    for word in prod_count.index.tolist():
        word_lst.append(d.check(word))
    prod_count['word'] = word_lst
    
    inv_frame = pd.DataFrame()
    oov_frame = pd.DataFrame()
    stat_frame = pd.DataFrame()
    # loop over the words
    for month in exp_count.columns.tolist():
        # select words existing in the exposure and training
        true_exp = exp_count[exp_count[month] > 0]
        true_prod = prod_count[prod_count[month] > 0]
        # get the corresponding words
        inv, oov, stat = get_intersect(true_exp[month], true_prod[[month,'word']],prod_type)
        # add month info
        inv['month'] = int(month)
        oov['month'] = int(month)
        # get prop by raw count
        inv_frame = pd.concat([inv_frame,inv])
        oov_frame = pd.concat([oov_frame,oov])
        stat_frame = pd.concat([stat_frame,stat])
    
    # sort by month
    inv_frame = inv_frame.sort_values(by='month')
    oov_frame = oov_frame.sort_values(by='month')
    stat_frame = stat_frame.sort_values(by='month')
    if not os.path.exists(output_path + prod_type):
        os.makedirs(output_path + prod_type)
    inv_frame.to_csv(output_path + prod_type + '/inv_token.csv')
    oov_frame.to_csv(output_path + prod_type +'/oov_token.csv')
    stat_frame.to_csv(output_path + prod_type +'/stat.csv')
    return inv_frame, oov_frame, stat_frame


def count_oov(exp_count_path,prod_count_path,output_path,prod_type):
    
    '''
    count oov tokens and oov token types
    
    input:
        - generation word count csv
        - exposure csv
        
    return 
        - oov tokens/words by months: token \ count \ month \ word(Boolean)
        - inv tokens/words by months: token \ count \ month \ word(Boolean)
        - stat frame with each month data

    '''
    
    def get_intersect(true_exp, true_prod,prod_type):
        
        '''
        get the intersected and oov tokens and their counts
        return: 2 dataframe with 4 columns: word, count, month, word(Boolean)
        '''
        true_exp.name = 'count'
        true_prod.columns.values[0] = 'count'
        # Overlapping elements
        overlapping_elements = set(true_exp.index).intersection(set(true_prod.index))
        # Elements in list A but not in list B
        difference_elements = set(true_prod.index).difference(set(true_exp.index))
        # select the subdataframe based on index
        inv = true_prod.loc[overlapping_elements]
        oov = true_prod.loc[difference_elements]
        inv['prod_type'] = prod_type
        oov['prod_type'] = prod_type
        # get the stat frame
        oov_token = oov['count'].sum()/true_prod['count'].sum()
        oov_word = oov[oov['word']==True]['count'].sum()/true_prod['count'].sum()
        oov_token_type = oov.shape[0]/true_prod.shape[0]
        oov_word_type = oov[oov['word']==True].shape[0]/true_prod.shape[0]
        inv_word = inv[inv['word']==True]['count'].sum()/true_prod['count'].sum()
        inv_token_type = inv.shape[0]/true_prod.shape[0]
        inv_word_type = inv[inv['word']==True].shape[0]/true_prod.shape[0]
        stat = pd.DataFrame([prod_type,int(month),oov_token,oov_word,oov_token_type,oov_word_type,
                             1-oov_token,inv_word,inv_token_type,inv_word_type]).T
        stat.columns = ['prod_type','month','oov_token','oov_word','oov_token_type','oov_word_type',
                             'inv_token','inv_word','inv_token_type','inv_word_type']
        return inv, oov, stat
    
    # check whether the generated tokens are true words
    exp_count = pd.read_csv(exp_count_path, index_col=0)
    prod_count = pd.read_csv(prod_count_path, index_col=0)
    # get the intersected columns of df
    prod_count = prod_count.reset_index().dropna(subset=['index']).set_index('index')
    selected_months = set(exp_count.columns).intersection(set(prod_count.columns))
    exp_count = exp_count[list(selected_months)]
    prod_count = prod_count[list(selected_months)]
    word_lst = []
    for word in prod_count.index.tolist():
        word_lst.append(d.check(word))
    prod_count['word'] = word_lst
    
    inv_frame = pd.DataFrame()
    oov_frame = pd.DataFrame()
    stat_frame = pd.DataFrame()
    # loop over the words
    for month in exp_count.columns.tolist():
        # select words existing in the exposure and training
        true_exp = exp_count[exp_count[month] > 0]
        true_prod = prod_count[prod_count[month] > 0]
        # get the corresponding words
        inv, oov, stat = get_intersect(true_exp[month], true_prod[[month,'word']],prod_type)
        # add month info
        inv['month'] = int(month)
        oov['month'] = int(month)
        # get prop by raw count
        inv_frame = pd.concat([inv_frame,inv])
        oov_frame = pd.concat([oov_frame,oov])
        stat_frame = pd.concat([stat_frame,stat])
    
    # sort by month
    inv_frame = inv_frame.sort_values(by='month')
    oov_frame = oov_frame.sort_values(by='month')
    stat_frame = stat_frame.sort_values(by='month')
    if not os.path.exists(output_path + prod_type):
        os.makedirs(output_path + prod_type)
    inv_frame.to_csv(output_path + prod_type + '/inv_token.csv')
    oov_frame.to_csv(output_path + prod_type +'/oov_token.csv')
    stat_frame.to_csv(output_path + prod_type +'/stat.csv')
    return inv_frame, oov_frame, stat_frame




exp_count_path = '/data/Machine_CDI/Lexical-benchmark_output/Final_scores/Model_eval/exp/count/raw_model.csv'
prod_count_path = '/data/Machine_CDI/Lexical-benchmark_output/Final_scores/Model_eval/exp/count/raw_adult1_36.csv'
output_path = '/data/Machine_CDI/Lexical-benchmark_output/Final_scores/Model_eval/exp/oov/'
prod_type = 'human_1_36'
prod_count_root = '/data/Machine_CDI/Lexical-benchmark_output/Final_scores/Model_eval/exp/count/'


month_dict = {'400':[4,8],'800':[9,18],'1600':[19,28],'3200':[29,36]}
root_path = '/data/Machine_CDI/Lexical-benchmark_data/exp/Audiobook/freq_by_train/'

# loop over the root folder
prod_type_lst = ['unprompted_0.3','unprompted_0.6','unprompted_1.5','unprompted_1.0']
inv_all = pd.DataFrame()
oov_all = pd.DataFrame()
stat_all = pd.DataFrame()
for prod_type in prod_type_lst:
    prod_count_path = prod_count_root + 'raw_' + prod_type + '6_36.csv'
    inv_frame, oov_frame, stat_frame = get_oov(exp_count_path,prod_count_path,output_path,prod_type)
    inv_all = pd.concat([inv_all,inv_frame])
    oov_all = pd.concat([oov_all,oov_frame])
    stat_all = pd.concat([stat_all,stat_frame])
    
stat_all.to_csv('/data/Machine_CDI/Lexical-benchmark_output/Final_scores/Model_eval/exp/oov/stat.csv')   
    
    
input_type = 'word'


def plot(stat_frame,input_type, system, label=None):
    
    #stat_frame = pd.read_csv(prop_frame_path)
    month_lst = [int(x) for x in stat_frame['month'].tolist()]
    prop_lst = [float(x) for x in stat_frame['oov_' + input_type].tolist()]
    sns.lineplot(month_lst,prop_lst, linewidth=3.5,label = label)
    plt.ylim(0,1)
    plt.xlabel('(Pseudo)age in month', fontsize=15)
    plt.ylabel('oov ' + input_type , fontsize=15)
    plt.title('OOV '+ input_type +' in ' + system, fontsize=15, fontweight='bold') 
  
    
    
prop_frame_path = '/data/Machine_CDI/Lexical-benchmark_output/Final_scores/Model_eval/exp/oov/human_6_36/stat.csv'
plot_CHILDES(prop_frame_path,'word','CHILDES')

plot_CHILDES(prop_frame_path,'token')

plot_CHILDES(prop_frame_path,'word_type')
plot_CHILDES(prop_frame_path,'token_type')


input_type = 'token_type'
prop_frame_path = '/data/Machine_CDI/Lexical-benchmark_output/Final_scores/Model_eval/exp/oov/'
temp_lst = ['unprompted_0.3','unprompted_0.6','unprompted_1.0','unprompted_1.5']
for temp in temp_lst:
    stat_frame = pd.read_csv(prop_frame_path + temp + '/stat.csv')
    plot(stat_frame,input_type, 'model', label=temp.split('_')[-1])









