# -*- coding: utf-8 -*-
"""
explore generated tokens

@author: Jing Liu
"""

import pandas as pd
import os
import collections
import enchant
import spacy
d = enchant.Dict("en_US")
# Load the English language model
nlp = spacy.load('en_core_web_sm')


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
        try:        
              
            # loop the parameter list
            for para in ['MONTH','PROMPT','BEAM','TOPK']:
                cleaned_frame.loc[i,para] = frame_all[i][para].tolist()[0]
                
        except:
            print(i)
            
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
                            data = pd.read_csv(root_path + '/' + month+ '/' + prompt_type + '/' + strategy + '/' + file)
                           
                            
                            
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
                                
                                if len(seq) > 0:
                                    seq_all.extend(seq)
                                
                            except:
                                print(file)
                                
    seq_lst = list(set(seq_all))
    
    seq_frame = match_seq(seq_lst,frame_all)
    
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
    
    
    
    word_lst.extend(['MONTH','PROMPT','BEAM','TOPK'])
    word_frame = seq_frame[word_lst]
    
    # reshape the lemma frame based onthe word_frame: basic info, lemma, total counts
    lemma_frame = seq_frame[['MONTH','PROMPT','BEAM','TOPK']]
    for lemma, words in lemma_dict.items():
        
        # Merge columns in the list by adding their values
        lemma_frame[lemma] = word_frame[words].sum(axis=1)
       
    
    return seq_frame, word_frame, lemma_frame





root_path = 'eval'
seq_frame, word_frame, lemma_frame = get_distr('eval')




def get_score(threshold,word_frame):
    
    '''
    get the score based on the threshold
    input:
        selected threshold
        dataframe with counts
        
    output:
        dataframe with the scores
    '''
    
    word_frame['MONTH']
    words = word_frame.drop(columns=['MONTH','PROMPT','BEAM','TOPK'])
    
    # Function to apply to each element
    def apply_threshold(value):
        if value > threshold:
            return 1
        else:
            return 0
    
    # Apply the function to all elements in the DataFrame
    words = words.applymap(apply_threshold)
   
    # append the file info and get fig in different conditions
    vocab_size_frame = word_frame[['MONTH','PROMPT','BEAM','TOPK']]
    vocab_size_frame['vocab_size']= words.sum(axis=1).tolist()

    return vocab_size_frame



# plot the curve in different conditions: only for the best fitness



# !!! TO DO: group the words by freq: esp OOV words(generated new words)


