#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate the reference entropy distr

@author: jliu
"""
import pandas as pd
import sys
import argparse
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from collections import Counter
'''
load evaluation model
'''

print('Loading evaluation model')
model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large', etc.
eval_model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
 


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='segment the generated sequences')
    
    parser.add_argument('--DataPath', type=str, default = '/scratch2/jliu/STELAWord/productive_vocab/reference/eval',
                        help='Path to the pretrained language model; even choose whether to use best checkpoint or not')
    
    parser.add_argument('--TrainPath', type=str, default = '/scratch2/jliu/STELAWord/productive_vocab/reference/eval',
                        help='Path to the pretrained language model; even choose whether to use best checkpoint or not')
    
    parser.add_argument('--ReferencePath', type=str, default = '/scratch2/jliu/STELAWord/productive_vocab/reference/eval',
                        help='Path to the reference prompt data')
    
    parser.add_argument('--mode', type=str, default = 'generated',
                        help='calculate sequence entropy; reference or generated')
    
    return parser.parse_args(argv)


def get_data(DataPath,ReferencePath):
    
    '''
    get the training data distr
    input: the .csv DataPath of the corresponding month
    output: .csv selected sequences with similar utterance length
    
    note that this can be accumulative; as the training data are accumulative as well; remove the length effect in this study
    '''
    # get the length distribution of the reference data
    prompt = pd.read_csv(ReferencePath)
    
    # select the corresponding length of data to ensure distr
    utt_len_lst = []
    n = 0
    while n < prompt.shape[0]:
        # to keep the similar length on the character level                                                                                
        utt_len = len(prompt['text_cleaned'].tolist()[n].replace(' ', ''))     
        utt_len_lst.append(utt_len)
        n += 1
    # select the corresponding utterances: note we need to keep the structure similar
    
    ref = pd.read_csv(ReferencePath,header = None)
    ref = pd.read_csv('train.txt',header = None)
    # add another column with the corresponding length
    train_len_lst = []
    n = 0
    
    while n < ref.shape[0]:
        # to keep the similar length on the character level                                                                                
        utt_temp = ref[0].tolist()[n].replace('|', '')
        utt_len = len(utt_temp.replace(' ', ''))  
        train_len_lst.append(utt_len)
        n += 1
    
    ref['length'] = train_len_lst
    # select the distribution with the corresponding length based on one list
    count_train = dict(Counter(train_len_lst))
    # round down the result so taht we can get the integers
    count_utt = dict(Counter(utt_len_lst))
    
    # align the results
    common_keys = set(count_train.keys()) & set(count_utt.keys())
    
    # get the times of the two different distribution
    common_values = {key: int(count_train[key]/count_utt[key]) for key in common_keys}
    
    # for the uncommon keys, truncate the rest of the lines and put in the freq dict
    add_keys = set(count_utt.keys()) - common_keys
    
    if len(add_keys) > 0:
        print('Please add more training set')           
    
    
    # seek the number of the sequences that might go with this condition
    expected_distr = {key: value * max(common_values.values()) for key, value in count_utt.items()}
    
    data_store = {}
    # as we always wish to truncate the largest one, just assure the left can be truncated from the larger distribution
    # reserve the lines with exactly similar length
    frame_repo = ref[~ref['length'].isin(expected_distr.keys())]
    material = pd.DataFrame()
    for key in expected_distr: 
        selected_frame = ref[ref['length']==key].iloc[:min(expected_distr[key],ref[ref['length']==key].shape[0])]
        material = pd.concat([material,selected_frame])
        
        # put the additional rows in the datastore
        if ref[ref['length']==key].shape[0] > expected_distr[key]:
            saved_frame = ref[ref['length']==key].iloc[expected_distr[key]:]
            frame_repo = pd.concat([frame_repo,saved_frame])
            
        # if not enough rows, record the left of the result in the form of a dictionary
        else:
            if expected_distr[key] - ref[ref['length']==key].shape[0] > 0:
                data_store[key] = expected_distr[key] - ref[ref['length']==key].shape[0]
                
    # make up for the distribution by truncating the existing sequences
    frame_repo = frame_repo.sort_values(by='length')
    start_index = 0
    
    for key in data_store: 
        # truncate the input sequences
        selected_frame = frame_repo.iloc[start_index:start_index + data_store[key]]
        start_index += data_store[key]
        
        # concatenate the rest of the dataframes
        
        
        
    return None
    



def calculate_entropy(eval_model,sentence):
    
    '''
    a gap between entropy based on the word level to the vhar level   # bpe tokens
    
    input: the single text string
    output: the calculated entropy score
    
    '''
    
    # Tokenize the sentence and convert to tensor
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    
        # Get the logits from the model
    with torch.no_grad():
            logits = eval_model(input_ids).logits[0]
    
    # Calculate the probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Calculate entropy
    entropy = -torch.sum(probabilities * torch.log2(probabilities))
    

    return entropy.item()





def main(argv):
    # Args parser
    args = parseArgs(argv)
    DataPath = args.DataPath
    mode = args.mode
    
    
    # loop all the genrated utterances
    # reduce the duplications: for files in the same folders or all the utterances? comparisons? 
    
    if mode == 'generated': 
        for month in os.listdir(DataPath):  
            
            for mode in os.listdir(DataPath + '/' + month):
                
                for stra in os.listdir(DataPath + '/' + month+ '/' + mode): 
                    
                    print('Write data for month: ' + month + ' and for mode:' + mode + 'with decoding strategy: ' + stra)
                    for file in tqdm(os.listdir(DataPath + '/' + month+ '/' + mode+ '/' + stra)):  
                        prompt  = pd.read_csv(DataPath + '/' + month+ '/' + mode+ '/' + stra + '/' + file)
                        
                        # only calcualte the results if there is no such column
                        if not 'LSTM_entropy' in prompt.columns:
                            entropy_lst = []
                            segmented_lst = []
                            n = 0
                            while n < prompt.shape[0]:
                                
                                if not 'LSTM_segmented' in prompt.columns:
                                    generated = prompt['LSTM_generated'].tolist()[n]
                                                            
                                    # segment generated sequences based on predicted word boundares
                                    try:
                                        segmented = generated.replace('|',' ')
                                    except:
                                        segmented = generated
                                        
                                    segmented_lst.append(segmented)
                                    
                                    
                                else:
                                    segmented = prompt['LSTM_segmented'].tolist()[n]
                                
                                
                                # calculate entropy
                                try:
                                    entropy = calculate_entropy(eval_model,segmented)
                                except:
                                    entropy = 0
                                    
                                entropy_lst.append(entropy)
                                n += 1
                            
                            if not 'LSTM_segmented' in prompt.columns:
                                # replace the original csv file
                                prompt['LSTM_segmented'] = segmented_lst
                                
                            prompt['LSTM_entropy'] = entropy_lst    
                            prompt.to_csv(DataPath + '/' + month+ '/' + mode+ '/' + stra + '/' + file)
                        
    else: 
        print('Wrong name!')
        
                    
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    