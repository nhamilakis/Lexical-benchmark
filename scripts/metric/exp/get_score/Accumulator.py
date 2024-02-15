'''
Accumulator model to examine children's input
'''

import os
import pandas as pd
import re
import math
import string
import collections


# preprocess files to get freq
filename_path = '/data/Machine_CDI/Lexical-benchmark_data/exp/Audiobook/'
text_path = '/data/Machine_CDI/Lexical-benchmark_data/train_phoneme/dataset/'
out_path = '/data/Machine_CDI/Lexical-benchmark_data/exp/Audiobook/freq_by_chunk/'
test_path = '/data/Machine_CDI/Lexical-benchmark_output/test_set/matched_set/char/bin_range_aligned/6_audiobook_aligned/'
lang = 'AE'
# get freq table recursively

def get_freq_table(lines):

    '''
    clean audiobook orthographic transcript
    input: 1. path to the .cha transcripts from CHILDES
           2. receotive or expressive vocab(related with the speaker)
    ouput: [the cleaned transcript],[the word list of the cleaned transcript]
    '''

    def get_freq(result):
        
        '''
        input: raw .txt file
        output: the freq dataframe with all the words and their raw freq
        '''
        # clean text and get the freq table
        frequencyDict = collections.Counter(result)  
        freq_lst = list(frequencyDict.values())
        word_lst = list(frequencyDict.keys())
        
        # get freq
        fre_table = pd.DataFrame([word_lst,freq_lst]).T
        col_Names=["Word", "Freq"]
        fre_table.columns = col_Names
        fre_table['Norm_freq'] = fre_table['Freq']/len(result)
        fre_table['Norm_freq_per_million'] = fre_table['Norm_freq']*1000000
        # get log_freq
        log_freq_lst = []
        for freq in freq_lst:
            log_freq = math.log10(freq)
            log_freq_lst.append(log_freq)
        fre_table['Log_freq'] = log_freq_lst
        
        # get logarithm of normalized word freq per million
        norm_log_freq_lst = []
        for freq in fre_table['Norm_freq_per_million'].tolist():
            norm_log_freq = math.log10(freq)
            norm_log_freq_lst.append(norm_log_freq)
        fre_table['Log_norm_freq_per_million'] = norm_log_freq_lst
        
        return fre_table

    # Remove empty lines using a list comprehension
    non_empty_lines_lst = [line for line in lines if line.strip()]
    word_lst = []
    for script in non_empty_lines_lst:
        # remove annotations
        translator = str.maketrans('', '', string.punctuation + string.digits)
        clean_string = script.translate(translator).lower()
        cleaned = re.sub(' +', ' ', clean_string.strip())
        # get the word lst
        words = cleaned.split(' ')
        word_lst.extend(words)
    
    fre_table = get_freq(word_lst)
    
    return fre_table



def count_chunk(filename_path,text_path,out_path):
    
    '''
    input: the filelst csv and the dir with .txt files
    '''
    # concatenate txt files
    
    file_frame = pd.read_csv(filename_path + 'Filename_chunk.csv')
    file_frame_grouped = file_frame.groupby('chunk')
    # get the cumulative count
    for chunk, file_frame_group in file_frame_grouped:
        line_lst = []
        for file in file_frame_group['file']:
            with open(text_path + file, 'r') as f:
                lines = f.readlines()
                line_lst.extend(lines)
                
        fre_table = get_freq_table(line_lst)    
        
        fre_table.to_csv(out_path + str(chunk) + '.csv')
    


def select_words(lang,out_path,filename_path):
   
    
    testset = pd.read_csv(test_path + 'machine_' + lang + '_exp.csv')
    
    # concatenate the results into one 
    
    freq_frame = pd.DataFrame()
    
    for file in os.listdir(out_path):
        freq_table = pd.read_csv(out_path + file)
        
        fre_lst = []
        for word in testset['word']:
            try:
                fre_lst.append(freq_table[freq_table['Word']==word]['Freq'].item())
            except:
                fre_lst.append(0)
           
        freq_frame[(int(file[:-4]) + 1) * 50/89] =  fre_lst
        
    freq_frame = freq_frame.reindex(sorted(freq_frame.columns), axis=1)
    freq_frame['word'] = testset['word']
    sel_frame = freq_frame.iloc[:,:-1]
    columns = freq_frame.columns[:-1]
    sel_frame = sel_frame.cumsum(axis=1)
    for col in columns.tolist():
        freq_frame[col] = sel_frame[col] 
    
    freq_frame.to_csv(filename_path + lang + '_freq.csv')
    



