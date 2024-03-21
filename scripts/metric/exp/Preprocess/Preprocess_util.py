#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
util func for preprocessing CHILDES dataset

@author: jliu
"""
import os
import zipfile
import pandas as pd
import re
from Rules import w2string, string2w, month_mode

def unzip(folder_path,extracted_dir):
    # Create the directory if it doesn't exist
    os.makedirs(extracted_dir, exist_ok=True)
    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Check if the file is a zip file
        if file_name.endswith('.zip'):
            # Open the zip file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Extract all contents to the specified directory
                zip_ref.extractall(extracted_dir)


def get_month(corpus_name, file_name, folder_path):
    '''
    get mode info based on the cha file name
    
    -directly from file name
    -from external lookup table
    '''

    def compute_month(file_name, mode, folder_path):

        if mode == 'month_first':
            #month = int(file_name[:-4][:2]) * 12 + int(file_name[:-4][2:4])
            month = int(file_name[:-8]) * 12 + int(file_name[-8:-6])
            
        if mode == 'month_final':
            digits = re.findall(r'\d', file_name)   # search from left to right
            month = int(digits[0]) * 10 + int(digits[1])

        if mode == 'folder_year':
            #month = int(folder_path)
            digits = re.findall(r'\d', folder_path.split(
                '/')[-1])   # search from left to right
            month = int(digits[0]) * 12

        if mode == 'folder_month':
            digits = re.findall(r'\d', folder_path.split(
                '/')[-1])   # search from left to right
            month = int(digits[0]) * 10 + int(digits[1])

        if mode == 'name':
            # read external lookup table
            age_frame = pd.read_csv(folder_path + '/age.csv', dtype={'Filename': str})
            
            selected_frame = age_frame[age_frame['Filename']
                                           == file_name[:-4]]
            # returnthe corresponding month
            month = int(selected_frame['Year'].tolist()[0]) * 12 + int(selected_frame['Month'].tolist()[0])
        
        if mode == 'year':
            # read external lookup table
            month = int(file_name[:-4][3:]) * 12

        if mode == 'week':
            # read external lookup table
            month = int(int(file_name[:-4]) / 4)

        if mode == 'fixed':
            # read external lookup table
            month = 11

        return month

    # retrieval the mode dictionary
    for mode, name in month_mode.items():
        if corpus_name in name:
            month = compute_month(file_name, mode, folder_path)

    return month



def clean_data(path):

    '''
    input: cha file
    return: cleaned csv file with columns: speaker, content
    '''
    
    def clean_speaker(text):
        return re.sub('[*:]', '', text)
    
    def clean_content(text):
        
        """
        clean the input annotated strings
            Remove non-word markers, event markers, and unintelligible speech 
            fix spelling
            remove punctuations and digits
        """
        
        # Define a regular expression pattern to match both the patterns to be removed and non-printable characters
        cleaned_text = re.sub(r'(@\w+\b|&=\w+\b|\[[^][]*?\]|\b(?:xxx|yyy|www)\b|[↗|\x00-\x1F\x7F-\x9F])','', text)
        
        # normalize spellings based on a predefined dict
        words = []
        for w in cleaned_text.split():
            # always lower-case
            w = w.lower()
            # fix spelling
            if w in w2string:
                w = w2string[w]
            w = w.replace('+', ' ').replace('_', ' ')    #split compound words
                # unfold abbreviations
            if w in string2w:
                w = string2w[w]
            words.append(w)

        script = ' '.join(words)
        # remove punctuations ang digits
        #cleaned_text = re.sub(r'[^a-zA-Z\s]+', '', script)
        cleaned_text = re.sub(r"([^a-zA-Z\s]+)|\'", '', script)
        
        # Replace consecutive spaces with a single space
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        # Remove leading and trailing spaces
        cleaned_text = cleaned_text.strip()
        return cleaned_text
    
    text_frame = pd.DataFrame()
    # clean annotation
    with open(path, encoding="utf8") as f:
        file = f.readlines()
        for script in file: 
            # only load utterances
            if script.startswith('*'):
                splitted = pd.DataFrame(script.replace('\n','').split('\t')).T
                text_frame = pd.concat([text_frame,splitted])
    # clean speaker and transcripts
    text_frame.columns = ['speaker','content']
    text_frame['speaker'] = text_frame['speaker'].apply(lambda x: clean_speaker(x))    
    text_frame['content'] = text_frame['content'].apply(lambda x: clean_content(x)) 
    # remove linew with only blanks
    cleaned_df = text_frame.dropna(axis=0, how='all').apply(lambda x: x.str.strip()).replace('', pd.NA).dropna(axis=0, how='all')
    cleaned_df = cleaned_df.dropna()
    return cleaned_df



def clean_cha(cha_path,month,corpus,lang):
    
    output_ele = cha_path.split('/')
    output_ele[6] = 'cleaned_transcript'
    output_path = '/'.join(output_ele[:-1])
    
    cleaned_frame = clean_data(cha_path)
    cleaned_frame['month'] = month
    cleaned_frame['corpus'] = corpus
    cleaned_frame['lang'] = lang
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_ele[-1] = output_ele[-1].split('.')[0] + '.csv'
    cleaned_frame.to_csv('/'.join(output_ele))
    print('Finished cleaning file: '+ item) 
    return cleaned_frame

# clean the most direct part: similar folder structure

def count_token(x):
    return len(x.split(' '))   

