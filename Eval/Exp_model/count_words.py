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
import matplotlib.pyplot as plt
import seaborn as sns


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
    
    cleaned_word_lst = ['MONTH','PROMPT','BEAM','TOPK','TEMP']
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
            for para in ['MONTH','PROMPT','BEAM','TOPK','TEMP']:
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
                                fre_table['TEMP'] = file.split('_')[1]
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
    
    
    
    word_lst.extend(['MONTH','PROMPT','BEAM','TOPK','TEMP'])
    word_frame = seq_frame[word_lst]
    
    # reshape the lemma frame based onthe word_frame: basic info, lemma, total counts
    lemma_frame = seq_frame[['MONTH','PROMPT','BEAM','TOPK','TEMP']]
    for lemma, words in lemma_dict.items():
        
        # Merge columns in the list by adding their values
        lemma_frame[lemma] = word_frame[words].sum(axis=1)
       
    
    return seq_frame, word_frame, lemma_frame



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
    words = word_frame.drop(columns=['MONTH','PROMPT','BEAM','TOPK','TEMP'])
    
    # Function to apply to each element
    def apply_threshold(value):
        if value > threshold:
            return 1
        else:
            return 0
    
    # Apply the function to all elements in the DataFrame
    words = words.applymap(apply_threshold)
   
    # append the file info and get fig in different conditions
    vocab_size_frame = word_frame[['MONTH','PROMPT','BEAM','TOPK','TEMP']]
    vocab_size_frame['vocab_size']= words.sum(axis=1).tolist()

    return vocab_size_frame


def plot_curve(month,vocab_size,y_label,title):
    
    '''
    input: a dataframe with varying sizes of temperature conditions
    '''
    # Plotting the line chart
    plt.figure(figsize=(8, 6))  # Optional: specify the figure size
    plt.plot(month,vocab_size, marker='o', linestyle='-')  # 'marker' defines the data point markers, 'linestyle' specifies the line style
    plt.xlabel(month)  # Label for x-axis
    plt.ylabel(y_label)  # Label for y-axis
    plt.title(title)  # Title of the plot
    plt.grid(True)  # Optional: add gridlines
    plt.show()
    
    
    
# plot the curve in different conditions: only for the best fitness

root_path = 'eval'
seq_frame, word_frame, lemma_frame = get_distr('eval')
threshold = 1
word_size_frame = get_score(threshold,word_frame)
lemma_size_frame = get_score(threshold,lemma_frame)

word_size_frame.to_csv('Vocab_size.csv')
lemma_size_frame.to_csv('Lemma_size.csv')
word_frame.to_csv('Vocab.csv')
lemma_frame.to_csv('Lemma.csv') 




n = 0 

decoding_lst = ['BEAM','TOPK']
prompt_lst = ['unprompted','prompted']
y_label = 'vocab size before lemmatization'

for prompt in prompt_lst:
    for decoding in decoding_lst: 
        # non-zero in the correspoonding column means that it is the chosen decoding type
        condition = (word_size_frame['PROMPT'] == prompt) & (word_size_frame[decoding] != '0')
        selected_frame_temp = word_size_frame[condition]
        for decode_para in list(set(selected_frame_temp[decoding].tolist())): 
            # sort the dataframe by number 
            
            selected_frame = selected_frame_temp[selected_frame_temp[decoding]==decode_para]
            selected_frame['MONTH'] = selected_frame['MONTH'].astype(int)

            # Sort the DataFrame based on the 'Column_Name'
            selected_frame = selected_frame.sort_values('MONTH')
            
            average_y_values = selected_frame.groupby('MONTH')['vocab_size'].mean().reset_index()
            plt.figure(figsize=(8, 6))  # Optional: specify the figure size
            sns.lineplot(data=average_y_values, x='MONTH', y='vocab_size', marker='o', ci=None)
            
            plt.ylim(20,120)
            plt.xlabel('month')  # Label for x-axis
            plt.ylabel(y_label)  # Label for y-axis (average)
            plt.title(prompt +': ' + decoding + '_' + decode_para)  # Title of the plot
            plt.grid(True)  # Optional: add gridlines
            plt.show()
            
            '''
            # plot the learning curve in different temp
            for temp in list(set(selected_frame['TEMP'].tolist())):
                final_frame = selected_frame[selected_frame['TEMP']==temp]    
                plot_curve(final_frame['MONTH'].tolist(),final_frame['vocab_size'].tolist(),y_label,  prompt +': ' + decoding)
            '''
            


# plot several vocab size trend 
# prompted topk


# unprompted topk


# prompted beam


# unprompted beam
# !!! TO DO: group the words by freq: esp OOV words(generated new words)




