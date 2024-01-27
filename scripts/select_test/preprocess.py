"""
Preprocess testset wwords:
human-CDI set:
    select content words

machine-CDI set:
    select wuggy subsets by the number of pseudo words pairs
"""

import pandas as pd
import string
import re
import spacy

nlp = spacy.load('en_core_web_sm')

def match_words(DataPath, fre_table, word_type):

    '''
    preprocess the test set

        input: word freq dataframe; CDI path
        output: selected word freq dataframe
    '''

    infants_data = pd.read_csv(DataPath)
    # remove annotations in wordbank
    words = infants_data['item_definition'].tolist()
    cleaned_lst = []
    for word in words:
        # remove punctuations
        translator = str.maketrans('', '', string.punctuation + string.digits)
        clean_string = word.translate(translator).lower()
        # remove annotations; problem: polysemies
        cleaned_word = re.sub(r"\([a-z]+\)", "", clean_string)
        cleaned_lst.append(cleaned_word)

    infants_data['word'] = cleaned_lst

    # merge dataframes based on columns 'Word' and 'words'
    df = pd.DataFrame()
    log_freq_lst = []
    norm_freq_lst = []
    n = 0
    while n < fre_table["Word"].shape[0]:
        selected_rows = infants_data[infants_data['word'] == fre_table["Word"].tolist()[n]]

        if selected_rows.shape[0] > 0:
            # for polysemies, only take the first meaning; OR
            clean_selected_rows = infants_data[infants_data['word'] == fre_table["Word"].tolist()[n]].head(1)
            log_freq_lst.append(fre_table['Log_norm_freq_per_million'].tolist()[n])
            norm_freq_lst.append(fre_table['Norm_freq_per_million'].tolist()[n])
            df = pd.concat([df, clean_selected_rows])
        n += 1

    df['Log_norm_freq_per_million'] = log_freq_lst
    df['Norm_freq_per_million'] = norm_freq_lst
    selected_words = df.sort_values('Log_norm_freq_per_million')

    # select open class words
    pos_all = []
    for word in selected_words['word']:
        doc = nlp(word)
        pos_lst = []
        for token in doc:
            pos_lst.append(token.pos_)
        pos_all.append(pos_lst[0])
    selected_words['POS'] = pos_all

    content_POS = ['ADJ', 'NOUN', 'VERB', 'ADV','PROPN']
    if word_type == 'all':
        selected_words = selected_words
    elif word_type == 'content':
        selected_words = selected_words[selected_words['POS'].isin(content_POS)]
    elif word_type == 'function':
        selected_words = selected_words[~selected_words['POS'].isin(content_POS)]

    return selected_words