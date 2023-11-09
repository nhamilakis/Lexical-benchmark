#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reference model hyperparameter testing in the case of prompted generation

compare the tested; put a wrapper func or bash code if you want to loop this recursively; outside the func

input: .csv dataframe with parents and children's utterances, segmented into turns
ouput: .csv with the generated tokens as an additional column

beam-search: unadded the para->n-gram blocking(do this later)
topk: k, temp
@author: jliu
"""

import pandas as pd
import torch
from generation_util import loadLSTMLMCheckpoint,word2char
import os 
import numpy as np
import sys
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import subprocess
from tqdm import tqdm

'''
load evaluation model
'''
model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large', etc.
eval_model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
 

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Generate tokens from decoder models')
    
    
    parser.add_argument('--ModelPath', type=str, default = '/scratch2/jliu/STELAWord/productive_vocab/reference/model/36/checkpoints',
                        help='Path to the pretrained language model; even choose whether to use best checkpoint or not')
    
    parser.add_argument('--DictPath', type=str, default = '/scratch2/jliu/STELAWord/productive_vocab/reference/data/36/bin_data/dict.txt',
                        help='Path to the dictionary data')
    
    parser.add_argument('--DataPath', type=str, default = '/scratch2/jliu/STELAWord/productive_vocab/reference_word/eval/Prompt_AE_sampled.csv',
                        help='Path to the parent-child utterance file')
    
    parser.add_argument('--OutputPath', type=str, default = '/scratch2/jliu/STELAWord/productive_vocab/reference/eval/36/prompted/',
                        help='Path to the generation output.')
    
    parser.add_argument('--mode', type=str, default = 'beam',
                        help='Different generation startegies: topk OR beam')
    
    parser.add_argument('--topk_lst', default = [1,5,20],
                        help='a list of top-k candidates for optimization')
    
    parser.add_argument('--temp_lst', default = [0.1,0.3,0.5,0.7,1],
                        help='a list of temperature parameters for optimization')
    
    parser.add_argument('--beam_lst', default = [1,5,20],    
                        help='a list of temperature parameters for optimization')
   
    return parser.parse_args(argv)



def run_command(command):
    subprocess.call(command, shell=True)
    

def generate_beam(seed_text,beam_width,temp,length,word_dict,task,generate_model):
    
    '''
    Beam search
    
    input: one string sequence
    output: generated string
    '''
    pad_idx = task.source_dictionary.pad()
    outputs = [[seed_text, 0.0]]  
    generated_tokens = []
    
    
    # try out the beam search 
    for i in range(length):
        
        all_candidates = []
        input_sequences = []
        
        # add more than one candidate to choose from
        for output in outputs: 
            
            generated_text, score = output
            # convert the generated text into a list
            sequences = generated_text.split(' ')
            
            '''
            iterate sequences
            basic generation is similar as greedy search
            start from this line   
            '''
            for sequence in sequences:    #!!!
                # Convert from string to list of units
                sentence_tokens = task.source_dictionary.encode_line(sequence, append_eos=False, add_if_not_exist=False).long()
                # Always generate similar tokens!! but similar tokens do not mean similar words 
                input_sequences.append(sentence_tokens)
            
            sequences_inputs = torch.nn.utils.rnn.pad_sequence(input_sequences, batch_first=True, padding_value=pad_idx).t()
            
            # Compute output & probabilities; shape of the last layer: (sequence_length, batch_size, vocabulary_size)
            
            with torch.no_grad():
                output_ts, _ = generate_model(sequences_inputs)
                
                # add the temperature para    
                scaled_logits = output_ts / temp     
                
                '''
                # note here we need to apply the softmax layer here in order to get the normalized prob distr
                sorted_values, sorted_indices = torch.topk(output_ts_updated, k=beam_width, dim=-1)
                
                '''
                softmax_output = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            
            sorted_values, sorted_indices = torch.topk(softmax_output, k=beam_width, dim=-1)
            '''
            sorted_values = sorted_values.to('cuda:0')
            sorted_indices = sorted_indices.to('cuda:0')
            '''
            
            indices = sorted_indices.tolist()[-1][-1]     # ends in this line
            
            # use word probability to rank the results
            word_probs = sorted_values.tolist()[-1][-1]     
            
            
            # prepare for the recursion 
            n = 0
            
            while n < len(indices):
                generated_words = ''
                generated_text_temp = generated_text
                index = indices[n]
                # expand different word sequences in this condition; to be updated on this level
                try:
                    
                    '''
                    note this should be re-started every time: for beam search specifically
                    '''
                    word = ' ' + list(word_dict)[index]    
                    generated_words += word
                    generated_text_temp += word
                    
                    # record the accumulated probability scores
                    new_score = score + np.log(word_probs[n])
                    all_candidates.append([generated_text_temp, new_score])
                    
                except:
                    pass
                n += 1
        # take sequence 
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        
        outputs = ordered[:beam_width]
    # get the most probable candidate and output the generated tokens
    generated_tokens = outputs[0][0].replace(seed_text + ' ', '')
   
    return generated_tokens
  

 
def generate_topk(seed_text,temp,topk,length,word_dict,task,generate_model):
    
    '''
    top-k decoding
    generate tokens based on external prompts that do not exist in the training dataset
    
    input: one string sequence
    output: generated string
    '''
    
    pad_idx = task.source_dictionary.pad()
    
    generated_tokens = []
    generated_words = ''
    generated_text = seed_text
    
    # try out the beam search 
    for i in range(length):
        
            # convert the generated text into a list
            sequences = generated_text.split(' ')
            
            '''
            iterate sequences
            basic generation is similar as greedy search
            start from this line   
            '''
            
            input_sequences = []
            for sequence in sequences:    
                # Convert from string to list of units
                sentence_tokens = task.source_dictionary.encode_line(sequence, append_eos=False, add_if_not_exist=False).long()
                # Always generate similar tokens!! but similar tokens do not mean similar words 
                input_sequences.append(sentence_tokens)
            
            sequences_inputs = torch.nn.utils.rnn.pad_sequence(input_sequences, batch_first=True, padding_value=pad_idx).t()
            
            # Compute output & probabilities; shape of the last layer: (sequence_length, batch_size, vocabulary_size)
            
            with torch.no_grad():
                output_ts, _ = generate_model(sequences_inputs)
                
                # add the temperature para    
                scaled_logits = output_ts / temp     
                
                '''
                # note here we need to apply the softmax layer here in order to get the normalized prob distr
                sorted_values, sorted_indices = torch.topk(output_ts_updated, k=beam_width, dim=-1)
                
                '''
                softmax_output = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            
            sorted_values, sorted_indices = torch.topk(softmax_output, k=topk, dim=-1)
            
            '''
            sorted_values = sorted_values.to('cuda:0')
            sorted_indices = sorted_indices.to('cuda:0')
            '''
            
            indices = sorted_indices.tolist()[-1][-1]     # ends in this line
            
            '''
            randomly sampling from 1 example top-k indices
            '''
            
            index = random.choice(indices)
            
            try:
                    
                # convert index into token
                word = ' ' + list(word_dict)[index]    
                generated_words += word
                generated_text += word
                
                
            except:
                pass
                
            # only get the first index
            generated_tokens.append(index)
  
    return generated_words



def calculate_entropy(eval_model,sentence):
    #!!!!
    
    '''
    a gap between entropy based on the word level to the vhar level   TO DO: check the entropy formula
    
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



'''
optimize hyperpara using Bayesian model

perform Bayesian search on hyperparameters

Q: the problem is GPT-2 and the training data are different in essence; whether it would be plausible enough 
at least we can count the number of words
'''



# go over all the model checkpoints recursively
def main(argv):
    # Args parser
    args = parseArgs(argv)
    
    
    # path to the training set(input) 
    DictPath = args.DictPath
   
    ModelPath = args.ModelPath
    pathLSTMCheckpoint = ModelPath + '/checkpoint_best.pt'
    # this is for model path
    DataPath = args.DataPath
    topk_lst = args.topk_lst
    temp_lst = args.temp_lst
    beam_lst = args.beam_lst
    OutputPath_temp = args.OutputPath
    mode = args.mode
    
    OutputPath = OutputPath_temp + mode
    # create a directory if there's no such path
    if not os.path.exists(OutputPath):
        os.makedirs(OutputPath)
    
    
    
    # Load input file
    print(f"Reading dictionary from {DictPath}")
    word_dict = {}
    with open(DictPath, 'r') as f:
        for line in f:
        
            non_empty_lines = [line for line in line.split("\n") if line.strip() != ""]
            cleaned = "\n".join(non_empty_lines)
            word_dict[cleaned.split(' ')[0]] = int(cleaned.split(' ')[1])
    print("Dictionary loaded!")

    # copy the dictionary to the model file
    run_command('cp ' + DictPath + ' ' + ModelPath + '/dict.txt')
    
    # Load LSTM model
    print("")
    print(f"Loading LSTM model from {pathLSTMCheckpoint}...")

    generate_model, task = loadLSTMLMCheckpoint(pathLSTMCheckpoint, ModelPath)
    generate_model.eval()
    print("Model loaded !")

   
    

    # generate tokens and calculate the entropy based on gpt-2
    print("")
    print("Loading prompt data from {pathQuantizedUnits}...")
    prompt = pd.read_csv(DataPath).head(2)
    
    
    if mode == 'top-k':
        for topk in tqdm(topk_lst):
            for temp in tqdm(temp_lst):
                
                print('Generating prompts with top-k: {} and with temperature: {}'.format(topk,temp))
                # get the generated new frame
                generated_lst = []
                #h_lst = []
                segmented_lst = []
                n = 0
                while n < prompt.shape[0]:
                    
                    # convert the word-level sequence into character level
                    
                    prompt_text = word2char(prompt['prompt_cleaned'].tolist()[n])
                    # chenge this into chracter length of children's utterances 
                    length = len(prompt['text_cleaned'].tolist()[n].replace(' ', ''))
                    
                    try:
                        # generate the prompt; note you need to turn this into the character sequence!
                        generated = generate_topk(prompt_text,temp,topk,length,word_dict,task,generate_model)
                                                
                        # segment generated sequences based on predicted word boundares
                        segmented = generated.replace('|',' ')
                        
                        #calculate the entropy
                        #h = calculate_entropy(eval_model,generated)
                    except:
                        generated = 'NA'
                        #h = 'NA'
                        segmented = 'NA'
                        
                    generated_lst.append(generated)
                    #h_lst.append(h)
                    segmented_lst.append(segmented)
                    n += 1
                # only preserve the selected data and the results; add an additional column to get the segemnted words
                prompt['LSTM_generated'] = generated_lst
                #prompt['LSTM_generated_h'] = h_lst
                
                # remove the error rows
                #prompt_cleaned = prompt[(prompt['LSTM_generated' ] != 'NA') & (prompt['LSTM_generated_h'] != 'NA')]
                prompt_cleaned = prompt[(prompt['LSTM_generated' ] != 'NA')]
                
                # save the file in file inthe naming convention: topk_temp_generated.csv
                prompt_cleaned.to_csv(OutputPath + '/' + str(topk) + '_' + str(temp) + '_generated.csv',escapechar='?')     # might need to deal with such info; or delete that
                print('Finished generation with top-k: {} and with temperature: {}'.format(topk,temp))
                
    
    
    # in the case of beam search
    else:
        for beam in tqdm(beam_lst):
            for temp in tqdm(temp_lst):
                
                print('Generating prompts with top-k: {} and with temperature: {}'.format(beam,temp))
                # get the generated new frame
                generated_lst = []
                #h_lst = []
                segmented_lst = []
                n = 0
                while n < prompt.shape[0]:
                    
                    # convert the word-level sequence into character level
                    
                    prompt_text = word2char(prompt['prompt_cleaned'].tolist()[n])
                    # chenge this into chracter length of children's utterances 
                    length = len(word2char(prompt['text_cleaned'].tolist()[n])) # convert to the char form to get the correct lenght
                        
                    try:
                        # generate the prompt; note you need to turn this into the character sequence!
                        generated = generate_beam(prompt_text,beam,temp,length,word_dict,task,generate_model)
                                    
                        # segment generated sequences based on predicted word boundares
                        segmented = generated.replace('|',' ')
                        
                        #calculate the entropy
                        #h = calculate_entropy(eval_model,generated)
                    except:
                        generated = 'NA'
                        #h = 'NA'
                        segmented = 'NA'
                        
                    generated_lst.append(generated)
                    #h_lst.append(h)
                    segmented_lst.append(segmented)
                    n += 1
                # only preserve the selected data and the results; add an additional column to get the segemnted words
                prompt['LSTM_generated'] = generated_lst
                #prompt['LSTM_generated_h'] = h_lst
                
                # remove the error rows
                #prompt_cleaned = prompt[(prompt['LSTM_generated' ] != 'NA') & (prompt['LSTM_generated_h'] != 'NA')]
                prompt_cleaned = prompt[(prompt['LSTM_generated' ] != 'NA')]
                
                # save the file in file inthe naming convention: topk_temp_generated.csv
                prompt_cleaned.to_csv(OutputPath + '/' + str(beam) + '_' + str(temp) + '_generated.csv',escapechar='?')     # might need to deal with such info; or delete that
                print('Finished generation with ' + mode + ': {} and with temperature: {}'.format(beam,temp))
                
    
    print("Finished generation!")

   
    
    
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    



