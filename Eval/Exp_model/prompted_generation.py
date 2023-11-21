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
from generation_util import loadLSTMLMCheckpoint,word2char,token_index,calculate_entropy,get_key_from_value
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
    
    parser.add_argument('--month', type=int, default = 1,
                        help= 'Dataset size that is equivalent to months of data')
    
    parser.add_argument('--ModelPath', type=str, default = '/scratch2/jliu/STELAWord/productive_vocab/reference/model',
                        help='Path to the pretrained language model; even choose whether to use best checkpoint or not')
    
    parser.add_argument('--DictPath', type=str, default = '/scratch2/jliu/STELAWord/productive_vocab/reference/data',
                        help='Path to the dictionary data')
    
    parser.add_argument('--DataPath', type=str, default = '/scratch2/jliu/STELAWord/productive_vocab/reference_word/eval/Prompt_AE_sampled.csv',
                        help='Path to the parent-child utterance file')
    
    parser.add_argument('--OutputPath', type=str, default = '/scratch2/jliu/STELAWord/productive_vocab/reference/eval',
                        help='Path to the generation output.')
    
    parser.add_argument('--mode', type=str, default = 'top-k',
                        help='Different generation startegies: topk OR beam')
    
    parser.add_argument('--topk_lst', default = [1],
                        help='a list of top-k candidates for optimization')
    
    parser.add_argument('--temp_lst', default = [0.1,0.3,0.5,0.7,1],
                        help='a list of temperature parameters for optimization')
    
    parser.add_argument('--beam_lst', default = [1,5],     # to be modified later!!!
                        help='a list of temperature parameters for optimization')
   
    return parser.parse_args(argv)



def run_command(command):
    subprocess.call(command, shell=True)
    


def generate_beam(seed_text,beam_width,temp,length,token_dict,task,generate_model):
    
    '''
    Beam search
    
    input: one string sequence; should be in the form of splitted characters with boundary markers
    output: generated string
    '''
    pad_idx = task.source_dictionary.pad()
    outputs = [[seed_text, 0.0]]  
    
    
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
                output_ts, _ = generate_model.forward(sequences_inputs)
                
                # add the temperature para    
                scaled_logits = output_ts / temp     
                
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
                    word = token_dict[index]    
                    generated_words += ' ' + word
                    generated_text_temp += ' ' + word
                    
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
    generated_all = outputs[0][0].replace(' ', '')
   
    return generated_all
  

 
def generate_topk(seed_text,temp,topk,length,token_dict,task,generate_model):
    
    '''
    top-k decoding
    generate tokens based on external prompts that do not exist in the training dataset
    
    input: one string sequence
    output: generated string
    '''
    
    pad_idx = task.source_dictionary.pad()
    
    generated_words = ''
    generated_text = seed_text
    
    # try out the beam search; chenge this into word length 
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
                output_ts, _ = generate_model.forward(sequences_inputs)
                
                # add the temperature para    
                scaled_logits = output_ts / temp     
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
                word = token_dict[index]   
                generated_words += ' ' + word
                generated_text += ' ' + word
                
                
            except:
                pass
                
    # remove the prompt 
    generated_words = generated_words.replace(" ", "")        
    return generated_words


seed_text = 't h i s | a | t e s t'
def generate_topk(seed_text,temp,topk,length,token_dict,task,generate_model):
    
    '''
    top-k decoding
    generate tokens based on external prompts that do not exist in the training dataset
    
    input: one string sequence
    output: generated string
    '''
    
    pad_idx = task.source_dictionary.pad()
    
    generated_words = ''
    generated_text = seed_text
    
    # keep word length similar as the reference length
    i = 0
    while i < length:

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
                output_ts, _ = generate_model.forward(sequences_inputs)
                
                # add the temperature para    
                scaled_logits = output_ts / temp     
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
            
            # only add i when 
            
            boundary_lst = []
            
                if value == '|' or value == '<s>' or value == '</s>':
                    i += 1
                
            try:
                    
                # convert index into token
                word = token_dict[index]   
                generated_words += ' ' + word
                generated_text += ' ' + word
                
                
            except:
                pass
                
    # remove the prompt 
    generated_words = generated_words.replace(" ", "")        
    return generated_words





# go over all the model checkpoints recursively
def main(argv):
    # Args parser
    args = parseArgs(argv)
    
    month = args.month
    # path to the training set(input) 
    DictPath = args.DictPath + '/' + month + '/bin_data/dict.txt' 
   
    ModelPath = args.ModelPath + '/' + month + '/checkpoints' 
    pathLSTMCheckpoint = ModelPath + '/checkpoint_best.pt'
    # this is for model path
    DataPath = args.DataPath
    topk_lst = args.topk_lst
    temp_lst = args.temp_lst
    beam_lst = args.beam_lst
    OutputPath_temp = args.OutputPath
    mode = args.mode
    
    OutputPath = OutputPath_temp + '/' + month + '/prompted/'  + mode
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

   
    # convert the word_dict into token dict
    token_dict = token_index(task,word_dict)
    print("Token dictionary loaded !")

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
                h_lst = []
                segmented_lst = []
                prob_lst = []
                n = 0
                while n < prompt.shape[0]:
                    
                    prompt_text = word2char(prompt['prompt_cleaned'].tolist()[n])
                    # change this into chracter length of children's utterances    Q: remove the punctuations? 
                    length = len(prompt['text_cleaned'].tolist()[n].replace(' ', ''))
                    
                    try:
                        # generate the prompt; note you need to turn this into the character sequence!
                        generated = generate_topk(prompt_text,temp,topk,length,token_dict,task,generate_model)
                         
                        # unify the format as to prepare for the input of GPT-2 calculation
                        # 1. segment generated sequences based on predicted word boundares and start/end of an utterance
                        segmented = generated.replace('</s>',' ')
                        segmented = segmented.replace('<s>',' ')
                        segmented = segmented.replace('|',' ')
                        # remove unknown token marker
                        segmented = segmented.replace('<unk>','')
                        segmented = segmented.replace('<pad>','')
                        # 2. lower case the generated tokens
                        segmented = segmented.lower()
                        # calculate the entropy and log prob
                        h, prob = calculate_entropy(eval_model,tokenizer,segmented)  
                        
                    except:
                        generated = 'NA'
                        h = 'NA'
                        prob = 'NA'
                        segmented = 'NA'
                        print('Something wrong with the generation')
                        
                    generated_lst.append(generated)
                    h_lst.append(h)
                    segmented_lst.append(segmented)
                    prob_lst.append(prob)
                    n += 1
                # add these results as additional columns
                prompt['LSTM_generated'] = generated_lst
                prompt['LSTM_generated_h'] = h_lst
                prompt['LSTM_segmented'] = segmented_lst
                prompt['LSTM_generated_prob'] = prob_lst
                # remove the error rows
                #prompt_cleaned = prompt[(prompt['LSTM_generated' ] != 'NA') & (prompt['LSTM_generated_h'] != 'NA')]
                prompt_cleaned = prompt[(prompt['LSTM_generated' ] != 'NA')]
                print('Finished generation with top-k: {} and with temperature: {}'.format(topk,temp))
                
    
    elif mode == 'beam':
        for beam in tqdm(beam_lst):
            for temp in tqdm(temp_lst):
                
                print('Generating prompts with beam: {} and with temperature: {}'.format(beam,temp))
                # get the generated new frame
                generated_lst = []
                h_lst = []
                segmented_lst = []
                prob_lst = []
                n = 0
                while n < prompt.shape[0]:
                    
                    prompt_text = word2char(prompt['prompt_cleaned'].tolist()[n])
                    # we keep the number of generations similar as CHILDES data 
                    length = len(prompt['text_cleaned'].tolist()[n].replace(' ', ''))
                    
                    try:
                        # generate the prompt; note you need to turn this into the character sequence!
                        generated = generate_beam(prompt_text,beam,temp,length,token_dict,task,generate_model)
                         
                        # unify the format as to prepare for the input of GPT-2 calculation
                        # 1. segment generated sequences based on predicted word boundares and start/end of an utterance
                        segmented = generated.replace('</s>',' ')
                        segmented = segmented.replace('<s>',' ')
                        segmented = segmented.replace('|',' ')
                        # remove unknown token marker
                        segmented = segmented.replace('<unk>','')
                        segmented = segmented.replace('<pad>','')
                        # 2. lower case the generated tokens
                        segmented = segmented.lower()
                        # calculate the entropy and log prob
                        h, prob = calculate_entropy(eval_model,tokenizer,segmented)
                        
                    except:
                        generated = 'NA'
                        h = 'NA'
                        prob = 'NA'
                        segmented = 'NA'
                        print('Something wrong with the generation')
                        
                    generated_lst.append(generated)
                    h_lst.append(h)
                    segmented_lst.append(segmented)
                    prob_lst.append(prob)
                    n += 1
                # add these results as additional columns
                prompt['LSTM_generated'] = generated_lst
                prompt['LSTM_generated_h'] = h_lst
                prompt['LSTM_segmented'] = segmented_lst
                prompt['LSTM_generated_prob'] = prob_lst
                # remove the error rows
                #prompt_cleaned = prompt[(prompt['LSTM_generated' ] != 'NA') & (prompt['LSTM_generated_h'] != 'NA')]
                prompt_cleaned = prompt[(prompt['LSTM_generated' ] != 'NA')]
                
                # save the file in file inthe naming convention: topk_temp_generated.csv
                prompt_cleaned.to_csv(OutputPath + '/' + str(beam) + '_' + str(temp) + '_generated.csv',escapechar='?')     # might need to deal with such info; or delete that
                print('Finished generation with beam: {} and with temperature: {}'.format(beam,temp))
       
    
    print("Finished generation!")

   
    
    
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    



