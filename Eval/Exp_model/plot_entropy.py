#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script to compare entropy distr of the train data and genrated prompts

@author: jliu
"""

import enchant
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   


dictionary = enchant.Dict("en_US")

data = pd.read_csv('eval/1/train_distr.csv')

# plot distr
h = data['entropy'].tolist()
prob = data['prob'].tolist()


def plot_distr(data,title):
    # Create a histogram using Seaborn
    #sns.histplot(data, kde=True, color='blue', stat='density')     # this will add the bard in the figure
    # Create the KDE plot without bars or data points
    sns.kdeplot(data, fill=True)
    # Add labels and title
    plt.xlabel(title)
    plt.ylabel('Frequency')
    plt.title(title + ' ditribution')
    
    # Show the plot
    plt.show()



'''
plot_distr(h,'entropy')
plot_distr(prob,'probability')
'''



def plot_distr(total_data,label_lst,title):

    n = 0
    while n < len(total_data):
    
        # Create the KDE plots without bars or data points
        sns.kdeplot(total_data[n], fill=True, label=label_lst[n])
        n += 1
    
    # Add labels and title
    plt.xlabel(title)
    plt.ylabel('Frequency')
    plt.title(title + ' ditribution')
    # Add legend
    plt.legend()
    # Show the plot
    plt.show()



# get the entropy distr for each type 
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
        for prompt_type in os.listdir(root_path + '/' + month):                     
            if not prompt_type.endswith('.csv') and not prompt_type.endswith('.ods'):                    
                for strategy in os.listdir(root_path + '/' + month+ '/' + prompt_type):                      
                    for file in os.listdir(root_path + '/' + month+ '/' + prompt_type+ '/' + strategy):
                        
                        # load decoding strategy information
                        strategy_lst.append(strategy)
                        strategy_lst.append(strategy)
                        month_lst.append(month)
                        data = pd.read_csv(root_path + '/' + month+ '/' + prompt_type + '/' + strategy + '/' + file)
                        temp_lst.append(file.split('_')[1])
                        
                        if strategy == 'beam':
                            beam_lst.append(file.split('_')[0])
                            topk_lst.append('0')
                            
                        if strategy == 'top-k':
                            topk_lst.append(file.split('_')[0])
                            beam_lst.append('0')
                            
                        
                        h_all.append(data['entropy'].tolist())
                        
info_frame = pd.DataFrame([month_lst,strategy_lst,beam_lst,topk_lst,temp_lst,h_all]).T

# rename the columns
info_frame.rename(columns = {0:'month', 1:'decoding', 2:'beam', 3:'topk', 4:'temp',5:'entropy'}, inplace = True)


# plot results in differetn conditions
# compare temperatures in the case of the same beam
para = 'temp'
target = info_frame[(info_frame['decoding']=='beam') & (info_frame['beam'])==1]




def count_words(root_path):
    
    '''
    get the number of generated words 
    input: root path of all the generated data
    output: infoframe: containing the number of the generated word types and in differetn parameter settings
            word count matrix for the occurrence of each word in the given generation method
            word freq matrix which checks each words appearance in the training set
            normzalized word count? -> we'll see this later
    '''
    month_lst = []
    vocab_size = []
    final_words =[]
    strategy_lst = []
    beam_lst = []
    topk_lst = []
    temp_lst = []
    # go over the generated files recursively
    root_path = 'eval'
    for month in os.listdir(root_path): 
        if not month.endswith('.csv') and not month.endswith('.ods'): 
            for prompt_type in os.listdir(root_path + '/' + month):                     
                if not prompt_type.endswith('.csv') and not prompt_type.endswith('.ods'):                    
                    for strategy in os.listdir(root_path + '/' + month+ '/' + prompt_type):                      
                        for file in os.listdir(root_path + '/' + month+ '/' + prompt_type+ '/' + strategy):
                            
                            # load decoding strategy information
                            strategy_lst.append(strategy)
                            month_lst.append(month)
                            data = pd.read_csv(root_path + '/' + month+ '/' + prompt_type + '/' + strategy + '/' + file)
                            temp_lst.append(file.split('_')[1])
                            
                            if strategy == 'beam':
                                beam_lst.append(file.split('_')[0])
                                topk_lst.append('0')
                                
                            if strategy == 'top-k':
                                topk_lst.append(file.split('_')[0])
                                beam_lst.append('0')
                                
                            sequence_lst = []
                            word_lst = []
                            # check whether it is a word or not? 
                            
                            for word in data['LSTM_generated'].tolist():
                                
                                try:
                                    #remove blank spaces between characters
                                    word = word.replace(" ", "")
                                except:    
                                    pass
                                
                                if type(word) == str:    
                                    # split the word sequence into a list
                                    for segmented_word in word.split('|'):
                                        sequence_lst.append(segmented_word)
                                        # check whether the genrated sequence is a word or not
                                        if len(segmented_word) > 0:
                                            if dictionary.check(segmented_word)==True: 
                                                word_lst.append(segmented_word)
                                          
                            vocab_size.append(len(set(word_lst)))
                            final_words.append(word_lst) 
                            
                            
    info_frame = pd.DataFrame([month_lst,strategy_lst,beam_lst,topk_lst,temp_lst,vocab_size,final_words]).T
    
    # rename the columns
    info_frame.rename(columns = {0:'month', 1:'decoding', 2:'beam', 3:'topk', 4:'temp',5:'vocab_size', 6:'words'}, inplace = True)
    return info_frame
                                
                                
info_frame = count_words('eval')                        
                        
info_frame.to_csv('eval/Reference_1.csv')                        
    