#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
util function for plotting entropy and calculating the fitness
@author: jliu
"""
import torch
from torch import nn
import torch.nn.functional as F
import enchant

dictionary = enchant.Dict("en_US")

def mean_KL(predicted,target):
    
    '''
    input: two lists of entropy 
        predicted: generated tokens; target: the reference ones, typically long and that's why we use the average value
    output: averaged KL distance
    
    '''
    
    def kl_divergence(reference,target):
    
        
        a = torch.tensor(reference)
        b = torch.tensor(target)
        
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        # input should be a distribution in the log space
        input_distr = F.log_softmax(a)
        # Sample a batch of distributions. Usually this would come from the dataset
        target = F.softmax(b)
        output = kl_loss(input_distr,target).item()
        return output
    # segment the predicted lists
    
    avg_dist_lst = []
    index = 0
    while index < len(target):
    
        segmented_target = target[index:index + len(predicted)] 
        try:
            dist = kl_divergence(predicted, segmented_target)
            avg_dist_lst.append(dist)
        except:
            pass
        index += len(predicted)
    
    avg_dist = sum(avg_dist_lst)/len(avg_dist_lst)
    return avg_dist


def plot_distr(reference_data, total_data,label_lst,x_label,title,prompt,month):
    
    '''
    plot the var dustribution
    input: 
        reference_data: a liost of the target training data
        tota_data: a list of the selected data in the given distr
        x_label: the chosen variable to investigate
        
    output: 
        the distr figure of the reference train set and the generated data 
    '''
    
    sns.kdeplot(reference_data, fill=False, label='train set')
    
    n = 0
    while n < len(total_data):
    
        # Create the KDE plots without bars or data points
        ax = sns.kdeplot(total_data[n], fill=False, label=label_lst[n])
        
        n += 1
    
    if x_label == 'entropy':
        for line in ax.lines:
            plt.xlim(0,18)
            plt.ylim(0,1.5)
            
    else:
        for line in ax.lines:
            plt.xlim(0,1)
            
            
    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel('Density')
    plt.title(prompt + ' generation, ' + title + ' in month ' + month)
    # Add legend
    plt.legend()
    # get high-quality figure
    plt.figure(figsize=(8, 6), dpi=800)
    plt.savefig(prompt + ' generation, ' + title + ' in month ' + month + '.png', dpi=800)
    # Show the plot
    plt.show()
    
    

def plot_single_distr(data,title):
    # Create a histogram using Seaborn
    #sns.histplot(data, kde=True, color='blue', stat='density')     # this will add the bard in the figure
    # Create the KDE plot without bars or data points
    sns.kdeplot(data, fill=True)
    # Add labels and title
    plt.xlabel(title)
    plt.ylabel('Density')
    plt.title(title + ' ditribution')
    
    # Show the plot
    plt.show()




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

