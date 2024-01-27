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
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import spacy

dictionary = enchant.Dict("en_US")
d = enchant.Dict("en_US")
# Load the English language model
nlp = spacy.load('en_core_web_sm')


'''
predicted = data['LSTM_generated_h'].tolist()
target = train_distr['entropy'].tolist()
'''

def mean_KL(predicted,target,gpu):
    
    '''
    input: two lists of entropy 
        predicted: generated tokens; target: the reference ones, typically long and that's why we use the average value
    output: averaged KL distance
    
    '''
    
    def kl_divergence(reference,target,gpu):
    
        if gpu:
            a = torch.tensor(reference).cuda()
            b = torch.tensor(target).cuda()
            
            kl_loss = nn.KLDivLoss(reduction="batchmean").cuda()
            # input should be a distribution in the log space
            input_distr = F.log_softmax(a).cuda()
            # Sample a batch of distributions. Usually this would come from the dataset
            target = F.softmax(b).cuda()
            output = kl_loss(input_distr,target).cuda()
            
        else:
            a = torch.tensor(reference)
            b = torch.tensor(target)
            
            kl_loss = nn.KLDivLoss(reduction="batchmean")
            # input should be a distribution in the log space
            input_distr = F.log_softmax(a)
            # Sample a batch of distributions. Usually this would come from the dataset
            target = F.softmax(b)
            output = kl_loss(input_distr,target)
            
            
        return output.item()
    # segment the predicted lists
    
    avg_dist_lst = []
    index = 0
    while index < len(target):
    
        segmented_target = target[index:index + len(predicted)] 
        
        try:
            dist = kl_divergence(predicted, segmented_target,gpu)
            avg_dist_lst.append(dist)
            
        except:
            print(index)
            
        index += len(predicted)
    
    avg_dist = sum(avg_dist_lst)/len(avg_dist_lst)
    return avg_dist



    
   
def plot_single_para(info_frame,reference,decoding,decoding_para,month,prompt,var):
    
    def plot_distr(reference_data, total_data,label_lst,x_label,title,prompt,month):
        
        '''
        plot the var dustribution     #!!! sort the labels
        input: 
            reference_data: a liost of the target training data
            tota_data: a list of the selected data in the given distr
            x_label: the chosen variable to investigate
            
        output: 
            the distr figure of the reference train set and the generated data 
        '''
        
        sns.set_style('whitegrid')
        sns.kdeplot(reference_data, fill=False, label='train set')
        
        n = 0
        while n < len(total_data):
        
            # Create the KDE plots without bars or data points
            ax = sns.kdeplot(total_data[n], fill=False, label=label_lst[n])
            
            n += 1
        
        if x_label == 'entropy':
            for line in ax.lines:
                plt.xlim(0,14)
                plt.ylim(0,1)
                
        else:
            for line in ax.lines:
                plt.xlim(0,19)
                plt.ylim(0,1)
                
        # Add labels and title
        plt.xlabel(x_label)
        plt.ylabel('Density')
        plt.title(prompt + ' generation, ' + title + ' in month ' + month)
        # Add legend
        plt.legend()
        # get high-quality figure
        plt.figure(figsize=(8, 6), dpi=800)
        plt.savefig('figure/' + prompt + ' generation, ' + title + ' in month ' + month + '.png', dpi=800)
        # Show the plot
        plt.show()
        
    
    target = info_frame[(info_frame['month']==month) & (info_frame['decoding']==decoding) & (info_frame['prompt']==prompt)
                        & (info_frame[decoding]==decoding_para)]
    
    # remove additional noise for random conditions
    if decoding == 'random':
        plot_distr(reference,target[var].tolist(),target['temp'].tolist(),var,decoding,prompt,month)

    else:
        plot_distr(reference,target[var].tolist(),target['temp'].tolist(),var,decoding + ': ' + decoding_para,prompt,month)



def plot_distance(target,var,decoding,prompt,month):  
    
        
    # Create a scatter plot with color mapping
   
    plt.scatter(target['temp'].tolist(),target[decoding].tolist(),c=target[var + '_dist'].tolist(), cmap='viridis')
   
   # Set labels and title
    plt.xlabel("temp")
    plt.ylabel(decoding)
    plt.title(prompt + ' generation, ' + decoding + ' in month ' + month)

    # Add a colorbar for reference
    plt.colorbar(label='KL divergence')

    # Show the plot
    plt.show()
    



def match_seq(cleaned_word_temp,frame_all):
    '''
    
    match the sequence with the genrated tokens
    '''
    
    cleaned_word_lst = ['MONTH','CHUNK','PROMPT','BEAM','TOPK','TEMP','TOPP','RANDOM','DECODING']
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
        try:        
              
            # loop the parameter list
            for para in ['MONTH','CHUNK','PROMPT','BEAM','TOPK','TEMP','TOPP','RANDOM','DECODING']:
                cleaned_frame.loc[i,para] = frame_all[i][para].tolist()[0]
                
        except:
            print(i)
            
        i += 1
        
    return cleaned_frame



def lemmatize(seq_lst):
    
    '''
    get words and lemmas from sequences respectively
    input: sequence list
    output: 1.word list; 2.lemma dict{lemma: words}
    '''
    # select the word from the dataframe
    
    lemma_dict = {}
    for seq in seq_lst:
        try: 
            
                # Process the word using spaCy
                doc = nlp(seq)
                # lemmatize the word 
                lemma = doc[0].lemma_
                
                if lemma in lemma_dict:
                    lemma_dict[lemma].append(seq)
                else:
                    lemma_dict[lemma] = [seq]
        except:
            pass

    return lemma_dict




def plot_word_count(seq_lst):
    
    '''
    get words and lemmas from sequences respectively
    input: sequence list
    output: 1.word list; 2.lemma dict{lemma: words}
    '''
    # select the word from the dataframe
    word_lst = []
    lemma_dict = {}
    for seq in seq_lst:
        try: 
            if d.check(seq) == True:
                word_lst.append(seq)
                # Process the word using spaCy
                doc = nlp(seq)
                # lemmatize the word 
                lemma = doc[0].lemma_
                
                if lemma in lemma_dict:
                    lemma_dict[lemma].append(seq)
                else:
                    lemma_dict[lemma] = [seq]
        except:
            pass

    return word_lst, lemma_dict



def get_score(target_frame,seq_frame,threshold):
    
    
    '''
    get the weighted score based on different frequency range
    
    '''
    
    overlapping_words = [col for col in target_frame['word'].tolist() if col in seq_frame.index.tolist()]
    # get the subdataframe
    selected_frame = seq_frame.loc[overlapping_words]
    
    # use weighted class to decrase the effect of the unbalanced dataset
    
    extra_word_lst = [element for element in target_frame['word'].tolist() if element not in overlapping_words]
    
    for word in extra_word_lst:
        selected_frame.loc[word] = [0] * selected_frame.shape[1]
        
    # get the score based on theshold
    score_frame_all = selected_frame.applymap(lambda x: 1 if x >= threshold else 0)
    
    avg_values = score_frame_all.mean()
    
    return score_frame_all, avg_values






def filter_words(word_frame, target_frame,by_freq):

    '''
    input: 1.word count dataframe 
           2.target frame with word and frequency columns

    Returns
            1. a list of target word count dataframe across different months
            2. name of different freq bands
    '''

    overlapping_words = [col for col in target_frame['word'].tolist() if col in word_frame.columns]
    
    
    # map the freq bands to each dataframe
    if by_freq:
        # get the subfrae of each freq bands
        target_frame_selected = target_frame[target_frame['word'].isin(overlapping_words)]
        # Grouping DataFrame by 'Category' column
        sub_dataframes = {}
        for freq in list(set(target_frame_selected['freq'].tolist())):
            selected_overlapping_words = target_frame_selected[target_frame_selected['freq']==freq]['word'].tolist()
            selected_overlapping_words.extend(['MONTH','CHUNK','PROMPT','BEAM','TOPK','TEMP','TOPP','RANDOM'])
            selected_frame = word_frame[selected_overlapping_words]
            sub_dataframes[freq] = selected_frame
        
    
    else:
        
        # append other info in the selected vocab
        overlapping_words.extend(['MONTH','CHUNK','PROMPT','BEAM','TOPK','TEMP','TOPP','RANDOM'])
        # Selecting existing columns based on the list of column names
        sub_dataframes = word_frame[overlapping_words]
    
    return sub_dataframes,len(overlapping_words)




def get_generation_length(prompt_path):
    # get total number of words
    prompt_file = pd.read_csv(prompt_path)     #'eval/Prompt_AE.csv'
    
    n = 0
    child_utt_num = 0
    while n < prompt_file.shape[0]:
        child_utt_num += len(prompt_file['child_cleaned'].tolist()[1].split())
        n += 1
    
    words_per_hour = 10000  
    child_utt_num/words_per_hour
    return child_utt_num


