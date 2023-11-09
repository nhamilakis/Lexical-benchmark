#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate the reference entropy distr

@author: jliu
"""
import torch
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
                        help='Path to the generated tokens by different trained models')
    
    parser.add_argument('--TrainPath', type=str, default = '/scratch2/jliu/STELAWord/productive_vocab/reference/data',
                        help='Root Path to the training data; different sizes in dufferent models')
    
    
    parser.add_argument('--mode', type=str, default = 'generated',
                        help='calculate sequence entropy; reference or generated')
    
    return parser.parse_args(argv)


def get_data1(DataPath,TrainPath,month):
    
    '''
    get the training data distr
    input: DataPath: the prompt path; ReferencePath: training dataset path; Output path: eval directory
    output: .csv selected sequences with similar utterance length
    
    note that this can be accumulative; as the training data are accumulative as well; remove the length effect in this study
    '''
    # get the length distribution of the reference data
    prompt = pd.read_csv(DataPath + '/Prompt_AE_sampled.csv')
    
    # select the corresponding length of data to ensure distr
    utt_len_lst = []
    n = 0
    while n < prompt.shape[0]:
        # to keep the similar length on the character level                                                                                
        utt_len = len(prompt['text_cleaned'].tolist()[n].replace(' ', ''))     
        utt_len_lst.append(utt_len)
        n += 1
    # select the corresponding utterances: note we need to keep the structure similar
    
    #ref = pd.read_csv('train.txt',header = None)
    ref_temp = pd.read_csv(TrainPath + '/' + month+ '/train.txt' ,header = None)
    
    # Note this line is temporary to get the results as soon as possible 
    ref_temp = ref_temp.iloc[800000:]
    
    # reanme column name of into the utt
    ref_temp.rename(columns={0: 'utt'}, inplace=True)
    
    # add another column with the corresponding length
    
    ref = pd.DataFrame()
    
    n = 0
    # concatenate the chosen lie just in case
    while n < ref_temp.shape[0]:
        # to keep the similar length on the character level 
        try:                                                                               
            utt_temp = ref_temp['utt'].tolist()[n].replace('|', '')
            utt_len = len(utt_temp.replace(' ', ''))  
            ref_temp['length'] = utt_len
            ref = pd.concat([ref,ref_temp])
            
        except:       
            # remove the corresponding rows
            print(n)
            
        n += 1
    
       
    train_len_lst = ref['length'].tolist()
    
    # select the distribution with the corresponding length based on one list
    count_train = dict(Counter(train_len_lst))
    # round down the result so taht we can get the integers
    count_utt = dict(Counter(utt_len_lst))
    
    # align the results
    common_keys = set(count_train.keys()) & set(count_utt.keys())
    
    # get the times of the two different distribution
    common_dict = {key: int(count_train[key]/count_utt[key]) for key in common_keys}
    
    # sample training set that correspond toi the utt distribution
    # multiply time; get the median
    
    # We use a smaller set to save time
    expected_distr = {key: value * sorted(list(common_dict.values()))[len(list(common_dict.values()))//30] for key, value in count_utt.items()}
    # look for the different sentence length
    add_keys = count_utt.keys() - common_keys
    
    if len(add_keys) > 0:
        #Please add more training set   
        for key in add_keys:
            expected_distr[key] = count_utt[key] * max(common_dict.values())
    
    
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
        # loop over the selected frame to truncate
        n = 0
        truncated_lst = []
        
        while n < selected_frame.shape[0]:
            
            truncated = selected_frame['utt'].tolist()[n][:key]
            truncated_lst.append(truncated)
            n += 1
            
        selected_frame['utt'] = truncated_lst
        selected_frame['length'] = key
        material = pd.concat([material,selected_frame])
    
    # print out the reference 
    material.to_csv(DataPath + '/' + month + '/train_distr.csv')   
    return material

def get_data(DataPath,TrainPath,month):
    
    '''
    get the training data distr
    input: DataPath: the prompt path; ReferencePath: training dataset path; Output path: eval directory
    output: .csv selected sequences with similar utterance length
    
    note that this can be accumulative; as the training data are accumulative as well; remove the length effect in this study
    '''
    
    ref_temp = pd.read_csv(TrainPath + '/' + month+ '/train.txt' ,header = None)
    
    # Note this line is temporary to get the results as soon as possible 
    material = ref_temp.iloc[:int(ref_temp.shape[0] * 0.1)]
    
    ref = pd.DataFrame()
    n = 0
    
    while n < material.shape[0]:
        try:
            ref_temp = material.loc[[n]]
            
            utt = material[0].tolist()[n].replace('|','').replace(' ','')
            ref_temp['utt'] = utt
            ref = pd.concat([ref,ref_temp])
        except:
            pass
        
        n+=1
    
    # print out the reference 
    ref.to_csv(DataPath + '/' + month + '/train_distr.csv')   
    return ref


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
    
    average_probability = torch.mean(probabilities)
    
    # normalize by the sequence length; otherwise this would be only the reflection of length distr
    norm_entropy = entropy/probabilities.shape[0] 
    return norm_entropy.item(),average_probability.item()
    


def main(argv):
    # Args parser
    args = parseArgs(argv)
    DataPath = args.DataPath
    TrainPath = args.TrainPath
    mode = args.mode
    
    
    # loop all the genrated utterances
    # reduce the duplications: for files in the same folders or all the utterances? comparisons? 
    
    if mode == 'generated': 
        for month in os.listdir(DataPath): 
            
            if not month.endswith('.csv') and not month.endswith('.ods'): 
                
                for mode in os.listdir(DataPath + '/' + month):
                    
                    if not mode.endswith('.csv') and not mode.endswith('.ods'): 
                        
                        for stra in os.listdir(DataPath + '/' + month+ '/' + mode): 
                            
                            if not stra.endswith('.csv') and not stra.endswith('.ods'): 
                                print('Write data for month: ' + month + ' and for mode:' + mode + 'with decoding strategy: ' + stra)
                                
                                for file in tqdm(os.listdir(DataPath + '/' + month+ '/' + mode+ '/' + stra)):  
                                    prompt  = pd.read_csv(DataPath + '/' + month+ '/' + mode+ '/' + stra + '/' + file)
                                    
                                    prompt_calculated = pd.DataFrame()   
                                    # calculate entropy
                                    n = 0
                                    while n < prompt.shape[0]:
                                        
                                        # only calculate those files that the given columns are not there 
                                        if 
                                        
                                        try:
                                            row_temp = prompt.iloc[[n]]
                                            # remove blank space and word boundary markers
                                            utt = row_temp['LSTM_generated'].tolist()[0].replace('|','').replace(' ','')    
                                            entropy, prob = calculate_entropy(eval_model,utt)
                                            row_temp['entropy_new'] = entropy 
                                            row_temp['prob_new'] = prob
                                            prompt_calculated = pd.concat([prompt_calculated,row_temp])
                                            
                                        except:
                                            pass
                                                
                                            
                                        n += 1
                                        
                                    prompt_calculated.to_csv(DataPath + '/' + month+ '/' + mode+ '/' + stra + '/' + file)
    
    #get the entropy of the reference utt
    else: 
       # loop though the training dataset and put it in the  
       #for month in os.listdir(TrainPath):  
       for month in ['36']:    
           print('Reading reference file of month ' + month)
           
           material = get_data(DataPath,TrainPath,month)
           #material = pd.read_csv(DataPath + '/' + month + '/train_distr.csv')
           
           material_final = pd.DataFrame()
           n = 0
           while n < material.shape[0]:
               try:
                   entropy, prob = calculate_entropy(eval_model,material['utt'].tolist()[n].replace('|','').replace(' ',''))
                   material_temp = material.loc[[n]]
                   material_temp['prob'] = prob
                   material_temp['entropy'] = entropy
                   material_final = pd.concat([material_final,material_temp])
               except:
                   pass
               n += 1
               
           
           
           material_final.to_csv(DataPath + '/' + month + '/train_distr.csv')
           print('Finish entropy and probability calculation ' + month)
           
           
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    