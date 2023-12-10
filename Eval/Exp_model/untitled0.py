#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count the numbwer of generated words from the model
@author: jliu
"""

import os
import pandas as pd
from plot_entropy_util import plot_single_para, plot_distance, match_seq,lemmatize
import collections
import argparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Investigate CHILDES corpus')
    
    parser.add_argument('--lang', type=str, default = 'AE',
                        help='langauges to test: AE, BE or FR')
    
    parser.add_argument('--TextPath', type=str, default = 'CHILDES',
                        help='root Path to the CHILDES transcripts; one of the variables to invetigate')
    
    parser.add_argument('--OutputPath', type=str, default = 'Output',
                        help='Path to the freq output.')
    
    parser.add_argument('--condition', type=str, default = 'recep',
                        help='types of vocab: recep or exp')
    
    parser.add_argument('--hour', type=int, default = 3,
                        help='the estimated number of hours per day')
    
    parser.add_argument('--word_per_hour', type=int, default = 10000,
                        help='the estimated number of words per hour')
    
    parser.add_argument('--threshold_range', type=list, default = [10],
                        help='threshold to decide knowing a productive word or not, one of the variable to invetigate')
    
    parser.add_argument('--eval_path', type=str, default = 'Human_CDI/',
                        help='path to the evaluation material; one of the variables to invetigate')
    
    return parser.parse_args(argv)






def load_data(root_path):
    
    '''
    count all the words in the generated tokens 
    
    input: the root directory containing all the generated files adn train reference
    output: 1.the info frame with all the generarted tokens
            2.the reference frame with an additional column of the month info
            3.vocab size frame with the seq word and lemma frequencies
    '''
    
    # load the rerference data
    frame_all = []
    seq_all = []
    month_lst = []
    prompt_lst = []
    h_all = []
    prob_all =[]
    strategy_lst = []
    beam_lst = []
    topk_lst = []
    topp_lst = []
    random_lst = []
    temp_lst = []
    directory_lst = []
    chunk_lst = []
    # go over the generated files recursively
    
    for month in os.listdir(root_path): 
        
        for chunk in os.listdir(root_path + '/' + month): 
            
            for prompt_type in os.listdir(root_path + '/' + month + '/' + chunk): 
                
                for strategy in os.listdir(root_path + '/' + month + '/' + chunk + '/' + prompt_type): 
                        
                    for file in os.listdir(root_path + '/' + month + '/' +  chunk+ '/' + prompt_type + '/' + strategy):
                            
                        # load decoding strategy information       
                        data = pd.read_csv(root_path + '/' + month + '/' +  chunk+ '/' + prompt_type + '/' + strategy + '/' + file)
                                    
                        try:
                                        
                         # count words
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
                             seq_all.extend(seq)
                                        
                                        
                             if strategy == 'beam':
                                 beam_lst.append(file.split('_')[0])
                                 topk_lst.append('0')
                                 topp_lst.append('0')
                                 random_lst.append('0')
                                 strategy_lst.append(strategy)
                                 fre_table['BEAM'] = file.split('_')[0]
                                 fre_table['TOPK'] ='0'
                                 fre_table['TOPP'] ='0'
                                 fre_table['RANDOM'] ='0'
                                            
                                            
                             elif strategy == 'sample_topk':
                                 topk_lst.append(file.split('_')[0])
                                 beam_lst.append('0')
                                 topp_lst.append('0')
                                 random_lst.append('0')
                                 strategy_lst.append(strategy.split('_')[1])
                                 fre_table['TOPK'] = file.split('_')[0]
                                 fre_table['BEAM'] ='0'
                                 fre_table['TOPP'] ='0'
                                 fre_table['RANDOM'] ='0'
                                            
                                            
                                            
                             elif strategy == 'sample_topp':
                                 topp_lst.append(file.split('_')[0])
                                 beam_lst.append('0')
                                 topk_lst.append('0')
                                 random_lst.append('0')
                                 strategy_lst.append(strategy.split('_')[1])
                                 fre_table['TOPP'] = file.split('_')[0]
                                 fre_table['BEAM'] ='0'
                                 fre_table['TOPK'] ='0'
                                 fre_table['RANDOM'] ='0'
                                            
                                            
                             elif strategy == 'sample_random':
                                 random_lst.append('1')
                                 topk_lst.append('0')
                                 beam_lst.append('0')
                                 topp_lst.append('0')
                                 strategy_lst.append(strategy.split('_')[1])
                                 fre_table['RANDOM'] = file.split('_')[0]
                                 fre_table['BEAM'] ='0'
                                 fre_table['TOPP'] ='0'
                                 fre_table['TOPK'] ='0'
                                        
                            # concatnete all the basic info regarding the genrated seq
                             fre_table['MONTH'] = month
                             fre_table['PROMPT'] = prompt_type
                             fre_table['TEMP'] = float(file.split('_')[1])
                             fre_table['CHUNK'] = chunk
                             prompt_lst.append(prompt_type)
                             month_lst.append(month)
                             temp_lst.append(float(file.split('_')[1]))
                             directory_lst.append(month+ '/' + prompt_type + '/' + strategy + '/' + file)
                             chunk_lst.append(chunk)
                             frame_all.append(fre_table)
                             
                             print('SUCCESS: ' + month+ '/' + prompt_type + '/' + strategy + '/' + file)    
                                    
                        except:
                             print('FAILURE: ' + month+ '/' + prompt_type + '/' + strategy + '/' + file)
                                
                    
    info_frame = pd.DataFrame([month_lst,chunk_lst,prompt_lst,strategy_lst,beam_lst,topk_lst,topp_lst,random_lst,temp_lst,directory_lst]).T
    
    # rename the columns
    info_frame.rename(columns = {0:'month', 1:'chunk', 2:'prompt',3:'decoding', 4:'beam', 5:'topk', 6:'topp',7:'random', 8:'temp',9:'entropy',10:'prob',11:'location'}, inplace = True)
   
    # remove the row with NA values
    info_frame = info_frame.dropna()
    info_frame = info_frame[(info_frame['random'] != '.') & (info_frame['topp'] != '.') & (info_frame['topk'] != '.')]
    
    
    # sort the result based on temperature to get more organized legend labels
    info_frame = info_frame.sort_values(by='temp', ascending=True)
    
    # # get word count and lemma count frames
    seq_lst = list(set(seq_all))
    
    seq_frame = match_seq(seq_lst,frame_all)
    
    word_lst, lemma_dict = lemmatize(seq_lst)
    
    word_lst.extend(['MONTH','CHUNK','PROMPT','BEAM','TOPK','TEMP','TOPP','RANDOM'])
    word_frame = seq_frame[word_lst]
    
    # reshape the lemma frame based onthe word_frame: basic info, lemma, total counts
    lemma_frame = seq_frame[['MONTH','CHUNK','PROMPT','BEAM','TOPK','TEMP','TOPP','RANDOM']]
    for lemma, words in lemma_dict.items():
        # Merge columns in the list by adding their values
        lemma_frame[lemma] = word_frame[words].sum(axis=1)
    
    return info_frame, seq_frame, word_frame, lemma_frame
   






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
    words = word_frame.drop(columns=['MONTH','CHUNK','PROMPT','BEAM','TOPK','TEMP','TOPP','RANDOM'])
    
    # Function to apply to each element
    def apply_threshold(value):
        if value > threshold:
            return 1
        else:
            return 0
    
    # Apply the function to all elements in the DataFrame
    words = words.applymap(apply_threshold)
   
    # append the file info and get fig in different conditions
    vocab_size_frame = word_frame[['MONTH','CHUNK','PROMPT','BEAM','TOPK','TEMP','TOPP','RANDOM']]
    vocab_size_frame['vocab_size']= words.sum(axis=1).tolist()

    return vocab_size_frame

root_path = 'eval'
var_lst = ['prob','entropy']
decoding_lst = ['topp']
prompt_lst = ['unprompted']
month_lst = ['1','3','12','36']
    

info_frame.to_csv('model_eval/info_frame.csv')
seq_frame.to_csv('model_eval/seq_frame.csv')
word_frame.to_csv('model_eval/word_frame.csv')
lemma_frame.to_csv('model_eval/lemma_frame.csv')


column_lst = ['MONTH','PROMPT','BEAM','TOPK','TEMP','TOPP','RANDOM']


threshold_lst = [1,2,3,4,5]
word_size_frame = get_score(threshold,word_frame)


# get and plot KL divergence 
# plot the vocab size curve



root_path = 'KL/KL_eval/'

kl_frame_all = pd.DataFrame()
lemma_frame_all = pd.DataFrame()
word_frame_all = pd.DataFrame()

for frame in os.listdir(root_path): 
    
    frame_temp = pd.read_csv(root_path + frame)
    
    
    if frame.startswith('Info'):

        kl_frame_all = pd.concat([kl_frame_all,frame_temp])
        kl_frame_all = kl_frame_all.drop_duplicates()
        

    elif frame.startswith('lemma'):
        lemma_frame_all = pd.concat([lemma_frame_all,frame_temp])
        lemma_frame_all = lemma_frame_all.drop_duplicates()
        
        
    elif frame.startswith('word'):
        word_frame_all = pd.concat([word_frame_all,frame_temp])
        word_frame_all = word_frame_all.drop_duplicates()
        
        
kl_frame_all = pd.read_csv('KL/KL_eval/Info_frame28.csv') 
lemma_frame_all = pd.read_csv('KL/KL_eval/lemma_frame28.csv') 
word_frame_all = pd.read_csv('KL/KL_eval/word_frame28.csv') 
kl_frame_all = kl_frame_all.drop_duplicates()
lemma_frame_all = lemma_frame_all.drop_duplicates()
word_frame_all = word_frame_all.drop_duplicates()

threshold = 1
lemma_size = get_score(threshold,lemma_frame_all)
word_size = get_score(threshold,word_frame_all)





var_lst = ['prob_dist','entropy_dist']

var_lst = ['prob_dist']
prompt_lst = ['unprompted']
month_lst = [1,3,12,36]
decoding = 'topp'
target_var = 'topp'



# use word count for evaluation 


def vocab_size(word_size,prompt_lst,month_lst,decoding_lst,vocab_type):
    
    decoding_dict = {'topk':'TOPK','beam':'BEAM','topp':'TOPP','random':'RANDOM'}
    
    sns.set_style('whitegrid')
    for prompt in prompt_lst:
              
        for decoding in decoding_lst:
                
                    target_frame = word_size[(word_size['PROMPT']==prompt) & (word_size['DECODING']==decoding)]
                    
                    for month in month_lst: 
                        # plot the difference for each month 
                        ax = sns.lineplot(data=target_frame[target_frame['MONTH'] == month], x=decoding_dict[decoding], y='vocab_size', label = str(month))
                    
                    
                    if prompt == 'unprompted':   
                        plt.ylim(0,400)  
                         
                        
                    elif prompt == 'prompted':  
                        
                        plt.ylim(50,250)  
                    
                    
                    plt.xlabel(decoding, fontsize=18)
                    plt.ylabel(vocab_type + '_size', fontsize=18)
                    plt.tick_params(axis='both', labelsize=14)
                    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='upper left',fontsize=15, title='month',title_fontsize=18)
                    plt.title(prompt + ' generation', fontsize=20)
                    plt.show()




prompt_lst = ['unprompted']
month_lst = ['50h','100h','200h']
decoding_lst = ['topp']
vocab_type = 'lemma'
word_size = lemma_size
vocab_size(word_size,prompt_lst,month_lst,decoding_lst,vocab_type)

var = 'prob_dist'
   

info_frame = kl_frame_all

def optimal_frame(info_frame,var,month_lst,prompt_lst,decoding_lst):
    
    '''
    get the dataframe with optimal conditions
    input: 
    output: the target dataframe
    
    '''
    frame_all = pd.DataFrame()
    for month in month_lst:
        for prompt in prompt_lst:
            for decoding in decoding_lst:
                selected_frame = info_frame[(info_frame['month']==month) & (info_frame['decoding']==decoding) & (info_frame['prompt']==prompt)]
                # get the row with the lowest target number 
                index_min = selected_frame[var].idxmin()
    
                # Get the row with the lowest value in the specified column
                target_row = selected_frame.loc[[index_min]]     
                # concatenate the given rows
                frame_all = pd.concat([target_row,frame_all])
    
    frame_all = frame_all.sort_values(by=var, ascending=True)
    return frame_all 

# get the optimal conditions with lowest KL divergence
prompt_lst = ['unprompted','prompted']
decoding_lst = ['topp','topk']
month_lst = [1,3,12,36]
var_lst = ['prob_dist','entropy_dist']

frame_dict = {}
for var in var_lst:
    frame = optimal_frame(kl_frame_all,var,month_lst,prompt_lst,decoding_lst)
    frame_dict[var] = frame
    
    
    
def main(argv):
    
    # Args parser
    args = parseArgs(argv)
    
    TextPath = args.TextPath
    condition = args.condition
    lang = args.lang
    OutputPath = args.OutputPath + '/' + lang + '/' + condition
    eval_path = args.eval_path + lang + '/' + condition
    hour = args.hour
    threshold_range = args.threshold_range
    word_per_hour = args.word_per_hour
    rmse_frame_all = pd.DataFrame()
    
    # step 1: load data and count words
    month_stat = load_data(TextPath,OutputPath,lang,condition)
    freq_frame = count_words(OutputPath,month_stat,eval_path,hour,word_per_hour)
    
    # step 2: get the score based on different thresholds
    for threshold in threshold_range:
    
        score_frame = get_score(freq_frame,OutputPath,threshold,hour)
             
        # plot the developmental bars and calculate the fitness
        print('Plotting the developmental bars')
             
        # compare and get the best combination of threshold two variables
        rmse = plot_curve(OutputPath,eval_path,score_frame,threshold,month_stat,condition)
        rmse_frame_temp = pd.DataFrame([threshold, rmse]).T
        rmse_frame = rmse_frame_temp.rename(columns={0: "Chunksize", 1: "threshold", 2: "rmse" })    
        rmse_frame_all = pd.concat([rmse_frame_all,rmse_frame])
        
       
    rmse_frame_all.to_csv(OutputPath + '/Scores/Fitness_All.csv')  
    
    

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    
    