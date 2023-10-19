# -*- coding: utf-8 -*-
"""
Receptive vocab

get freq of CDI receptive vocab from CHILDES parents' utterances
        input: 1.CHILDES transcript; 2.CDI data
        output: all are in the same folder named as lang_condition 
        1. all the selected transcripts by interlocutor in CHILDES
        2. the freq list of all the words in the selected transcripts 
        3. the freq list of all the words in the selected transcripts with the overlapping of CDI data

"""
import string
import os
import pandas as pd
import re
import sys
import spacy
import argparse
from util import clean_text, get_freq

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Investigate CHILDES corpus')
    
    parser.add_argument('--TextPath', type=str, default = 'CHILDES',
                        help='root Path to the CHILDES transcripts')
    
    parser.add_argument('--DataPath', type=str, default = 'CDI',
                        help='root Path to the CDI data')
    
    parser.add_argument('--OutputPath', type=str, default = 'Output',
                        help='Path to the freq output.')
    
    parser.add_argument('--lang', type=str, default = 'BE',
                        help='AE/BE/FR')
    
    parser.add_argument('--condition', type=str, default = 'production',
                        help='comprehension mode for caregivers; production mode for children output')
    
    parser.add_argument('--word_type', type=str, default = 'content',
                        help='word_type for the selected words: content/function/all ')
    
    parser.add_argument('--freq_bands', type=int, default = 3,
                        help='number of freq bands to chunk into')
    
    return parser.parse_args(argv)

    
def load_data(TextPath,OutputPath,lang,condition):
    
    '''
    # Load the .cha file
    # three folder structures
    output: the concatnated file; number of sentences
    '''
    
    word_lst = []
    final_lst = [] 
    for file in os.listdir(TextPath + '/' + lang):
        foldername = os.fsdecode(file)
        
        for text in os.listdir(TextPath + '/' + lang + '/' + foldername):
            text_name = os.fsdecode(text)
            CHApath = TextPath + '/' + lang + '/' + foldername + '/' + text_name
            final, word = clean_text(CHApath,condition)
            if len(final) > 0:
                final_lst.append(final)
                word_lst.append(word)
    
    # flatten the content and word list
    text_list = [item for sublist in final_lst for item in sublist]
    word_list = [item for sublist in word_lst for item in sublist]
    # print out the concatenated file
    output_dir = OutputPath + '/' + lang + '/' + condition 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_dir + '/transcript.txt', 'w', encoding="utf8") as f:
        for line in text_list:
            f.write(line + '\n')  
            
    return word_list, len(text_list), len(word_list)




def count_words(result,OutputPath,lang,condition):
    
    '''
    input: raw word list extracted from the transcripts 
    output: the freq dataframe with all the words adn their raw freq
    '''
    
    fre_table = get_freq(result)    
    # print out the concatenated file
    output_dir = OutputPath + '/' + lang + '/' + condition 
    fre_table.to_csv(output_dir + '/Freq_all.csv')
    return fre_table



def match_words(DataPath,output_dir, lang, fre_table, word_type):
    
    '''
    input: word freq dataframe; CDI path 
    output: selected word freq dataframe 
    '''
    
    infants_data = pd.read_csv(DataPath)
    # remove annotations in wordbank
    words = infants_data['item_definition'].tolist()
    cleaned_lst = []
    for word in words:
        # remove punctuations
        translator = str.maketrans('', '', string.punctuation+ string.digits)
        clean_string = word.translate(translator).lower()
        # remove annotations; problem: polysemies
        cleaned_word = re.sub(r"\([a-z]+\)", "",clean_string)
        
        cleaned_lst.append(cleaned_word)
    
    infants_data['words'] = cleaned_lst
    
    # merge dataframes based on columns 'Word' and 'words' 
    df = pd.DataFrame()
    log_freq_lst = []
    norm_freq_lst = []
    n = 0
    while n < fre_table["Word"].shape[0]:
        selected_rows = infants_data[infants_data['words'] == fre_table["Word"].tolist()[n]] 
        
        if selected_rows.shape[0] > 0:
            # for polysemies, only take the first meaning; OR 
            clean_selected_rows = infants_data[infants_data['words'] == fre_table["Word"].tolist()[n]].head(1)
            
            log_freq_lst.append(fre_table['Log_norm_freq_per_million'].tolist()[n])
            norm_freq_lst.append(fre_table['Norm_freq_per_million'].tolist()[n])
            df = pd.concat([df, clean_selected_rows])    
        n += 1
        
    df['Log_norm_freq_per_million'] = log_freq_lst
    df['Norm_freq_per_million'] = norm_freq_lst
    selected_words = df.sort_values('Log_norm_freq_per_million')
    
    # select words based on word type
    # get content words
    if lang == 'AE' or lang == 'BE': 
        nlp = spacy.load('en_core_web_sm')
    elif lang == 'FR': 
        nlp = spacy.load('fr_core_news_sm')    
        
    pos_all = []
    for word in selected_words['words']:     
        doc = nlp(word)
        pos_lst = []
        for token in doc:
            pos_lst.append(token.pos_)
        pos_all.append(pos_lst[0])
    selected_words['POS'] = pos_all
    
    func_POS = ['PRON','SCONJ','CONJ','CCONJ','DET','AUX', 'INTJ','PART']
    if word_type == 'all':
        selected_words = selected_words
    elif word_type == 'content':
        selected_words = selected_words[~selected_words['POS'].isin(func_POS)]
    elif word_type == 'function':
        selected_words = selected_words[selected_words['POS'].isin(func_POS)]
    
    # annotatie lists
    if not os.path.exists(output_dir):        
        os.makedirs(output_dir)
        
    selected_words.to_csv(output_dir + '/Freq_selected_' + word_type + '.csv')
    
    return selected_words

# chunk into different frequency bins
"""
use log freq bins to create different subgroups
"""        

def create_group(lang, mode, selected_words, trial,group_num):
    
    # output the corresponding indices
    final_index_temp = []
    fre_band = []
    
    for num in range(group_num):
        fre_band.append(str(2**num))
        
    for i in range(group_num):  
        final_index_temp.append([fre_band[i]] * len(trial[i]))
        
    # flatten the list
    final_index_lst = [item for sublist in final_index_temp for item in sublist]    
    selected_words['Group'] = final_index_lst
    selected_words.to_csv(lang + '_' + mode + '_' + 'freq_selected.csv')
    return selected_words




def main(argv):
    
    # Args parser
    args = parseArgs(argv)
    
    OutputPath = args.OutputPath 
    TextPath = args.TextPath
    lang = args.lang
    condition = args.condition
    DataPath = args.DataPath
    word_type = args.word_type
    
    DataFile = DataPath + '/' + lang + '/' + condition 
    
    stat = {}
    output = OutputPath + '/' + lang + '/' + condition 
    # step 1: concatenate the selected transcripts
    
    print('Start concatenating files')
    words, sent_num, word_num = load_data(TextPath,OutputPath,lang,condition)
    print('Finished concatenating files')
        
    # step 2: output all the word count
    if os.path.exists(output + '/Freq_all.csv'):        
        print('The frequency file already exists, skip')
        #fre_table = pd.read_csv(output + '/Freq_all.csv')
        fre_table = count_words(words,OutputPath,lang,condition)
    else: 
        print('Start counting words')
        fre_table = count_words(words,OutputPath,lang,condition)
        print('Finished counting words')
    
    # step 3: output all the selected word count
    print('Start matching CHILDES and CDI data')
    for file in os.listdir(DataFile):
        
        file_name = os.fsdecode(file)
        print('Selecting CHILDES words from ' + file_name)
        
        output_dir = OutputPath + '/' + lang + '/' + condition + '/' + file_name.split('.')[0]
        
        selected_words = match_words(DataFile + '/' + file_name,output_dir, lang, fre_table, word_type)    
        stat['selected ' + word_type +' words_' + file_name.split('.')[0]] = selected_words.shape[0]
    
        
    '''
    step 4: get statistics of the transcripts and word distribution
    for transcript: # of words and sentences
    selected words: # of the selected words
    '''
    
    stat['number of sentences'] = sent_num
    stat['word counts'] = word_num
    stat['number of word types'] = fre_table.shape[0]
    
    stat_df = pd.DataFrame.from_dict(stat, orient="index")
    stat_df.to_csv(output + '/stat.csv')
    
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)





