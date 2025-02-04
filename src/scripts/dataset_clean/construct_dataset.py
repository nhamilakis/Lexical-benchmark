import pandas as pd
import os
import re
import string
import numpy as np
from tqdm import tqdm



def cut_sent(sentences, target_words):
    # Count words in each sentence
    word_counts = [len(sentence.split()) for sentence in sentences]

    # Get cumulative word count
    cumulative_words = 0
    for i, count in enumerate(word_counts):
        cumulative_words += count
        # If we exceed or match target, check which is closer
        if cumulative_words >= target_words:
            # Check if including or excluding this sentence is closer to target
            with_current = cumulative_words
            without_current = cumulative_words - count

            # Compare distances to target
            if abs(with_current - target_words) <= abs(without_current - target_words):
                return (sentences[:i+1], sentences[i+1:], with_current)
            else:
                return (sentences[:i], sentences[i:], without_current)

    # If we never reached the target, return the full list and empty list
    return (sentences, [], cumulative_words)



# cut into different 

def split_by_wordcount(sentences, target_words):
    word_counts = [len(sentence.split()) for sentence in sentences]
    cumulative_words = 0
    splits = []
    current_split = []
    
    for i, count in enumerate(word_counts):
        cumulative_words += count
        if cumulative_words >= target_words:
            # Check which split point is closer to target
            with_current = cumulative_words
            without_current = cumulative_words - count
            
            if abs(with_current - target_words) <= abs(without_current - target_words):
                current_split = sentences[:i+1]
                remaining = sentences[i+1:]
            else:
                current_split = sentences[:i]
                remaining = sentences[i:]
                
            splits.append(current_split)
            
            # Reset for next split if there are remaining sentences
            if remaining:
                sentences = remaining
                cumulative_words = 0
                
    # Add any remaining sentences as the last split
    if sentences:
        splits.append(sentences)
    
    return splits

def merge(list_of_sentence_lists, k):
    # First, split each list into k parts
    all_splits = []
    for sentences in list_of_sentence_lists:
        total_words = sum(len(sent.split()) for sent in sentences)
        target_per_split = total_words // k
        splits = split_by_wordcount(sentences, target_per_split)
        
        # Ensure we have exactly k splits by adding empty lists if necessary
        while len(splits) < k:
            splits.append([])
            
        all_splits.append(splits)
    
    # Merge corresponding splits
    merged_splits = []
    for i in range(k):
        merged = []
        for splits in all_splits:
            if i < len(splits):
                merged.extend(splits[i])
        merged_splits.append(merged)
    
    return merged_splits

def get_len(text:str)->int:
    return len(text.split())

def divide_df(df, column, target_sum):
    """
    Divides a dataframe into subdataframes such that the sum of the specified column 
    in each subdataframe is close to the given target sum using cumsum for efficiency.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    column (str): The column to balance by sum.
    target_sum (float): The target sum for each subdataframe.

    Returns:
    list: A list of subdataframes.
    """
    # Sort the dataframe by the column in descending order
    df_sorted = df.sort_values(by=column, ascending=False).reset_index(drop=True)

    # Calculate cumulative sum and assign group IDs
    df_sorted['cumsum'] = df_sorted[column].cumsum()
    df_sorted['group'] = (df_sorted['cumsum'] // target_sum).astype(int)

    # Group rows into subdataframes
    subdataframes = [group.drop(columns=['cumsum', 'group']) for _, group in df_sorted.groupby('group')]

    return subdataframes

transcript_mode = 'train'
debug = False
data_root = '/scratch1/projects/lexical-benchmark/v2/datasets/ChildRealistic'
added_root = f'{data_root}/src/preprocessed/txt/EN'
out_root = f'{data_root}/txt/EN/50h'



# read and load files
# count words respectively
n = 60
lists = []
for file in tqdm(os.listdir(added_root)):
    if file.endswith(f'{transcript_mode }.preprocessed'):
        print(file)
        with open(f'{added_root}/{file}','r') as f:
            data = f.readlines()
            if debug:
                print('Entering the debugging mode!') 
                data = data[:600]
            
            # convert into a df
            df = pd.DataFrame([data]).T
            df.columns = ['text']
            # perprocess the text
            df['word_count'] = df['text'].apply(get_len)
            # divide the dataframe into the given number of subdf
            target_sum = int(df['word_count'].sum()/n)
            df_lst = divide_df(df, 'word_count', target_sum)
            # only get the target number of df
            print('Finished the target sum')
            print(len(df_lst))
            k = 0
            for subdf in tqdm(df_lst):
                print({k:subdf['word_count'].sum()})
                k += 1

        lists.append(df_lst)
        print({file:df['word_count'].sum()})

print('Finished concatenating files')
print(len(lists))


# segment into different chunks
filename = 'transcription.txt'
n = 0
while n < 60:
    # get the df from different chunks
    chunk_df = pd.DataFrame()
    for df_lst in lists:
        subdf = df_lst[n]
        chunk_df = pd.concat([chunk_df,subdf])
    print({n:chunk_df['word_count'].sum()})
    
    # save the results to the corresponding location
    chunk_name = f"{n:02d}"
    out_path = f'{out_root}/{str(chunk_name)}'
    os.makedirs(out_path, exist_ok=True)
    # save as the text file
    with open(f'{out_path}/{filename}', 'w') as file:
        for text in chunk_df['text']:
            file.write(str(text))

    print(f'Having saved the file to {out_path}')
    n += 1

