#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reference model hyperparameter testing using CHILDES parents' prompts

input: .csv dataframe with parents and children's utterances, segmented into turns
ouput: .csv with the generated tokens as an additional column

beam-search: unadded the para->n-gram blocking(do this later)
topk: k, temp
@author: jliu
"""

import pandas as pd
import torch
from generation_util import *
import os 
import numpy as np
import sys
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Generate tokens from decoder models')
    
    
    parser.add_argument('--month', type=str, default = '1',
                        help= 'Dataset size that is equivalent to months of data')
    
    parser.add_argument('--ModelPath', type=str, default = 'model',
                        help='Path to the pretrained language model; even choose whether to use best checkpoint or not')
    
    parser.add_argument('--DictPath', type=str, default = 'data',
                        help='Path to the dictionary data')
    
    parser.add_argument('--DataPath', type=str, default = 'eval/Prompt_AE_sampled.csv',
                        help='Path to the parent-child utterance file')
    
    parser.add_argument('--OutputPath', type=str, default = 'eval_trial',
                        help='Path to the generation output.')
    
    parser.add_argument('--prompt_mode', type=str, default = 'unprompted',
                        help='promtped to unprompted conditions')
    
    parser.add_argument('--mode', type=str, default = 'beam',
                        help='Different generation startegies: sample_topk,sample_topp, sample_random OR beam')
    
    parser.add_argument('--sample_lst', default = '1',
                        help='a list of top-k or top_p or beam candidates')
    
    parser.add_argument('--temp_lst', default = '0.1,0.3',
                        help='a list of temperature parameters for optimization')
    
    parser.add_argument('--gpu', type=bool, default = True,
                        help= 'whether to use gpu')
    
    parser.add_argument('--boundary_lst', default = ['<s>', '</s>', '|'],     
                        help='a list of potential word boundaries')
    
    parser.add_argument('--target_tokens', default = ['|','e','t','a','h','o','n','i','s','r','d','l','w', 'u','m','g','c','f','y','b','p','k','v','j','x','q','z'],     
                        help='a list of target tokens')
    
    return parser.parse_args(argv)




def generate_beam(seed_text,beam_width,temp,length,token_dict,boundary_lst,target_tokens,task,generate_model,gpu):
    
    '''
    Beam search
    
    input: one string sequence; should be in the form of splitted characters with boundary markers
    output: generated string
    '''
    
    if gpu:
        generate_model = generate_model.cuda()
        
    target_tokens.extend(boundary_lst)
    # get indices to be filtered
    filtered_indices = [key for key, value in token_dict.items() if value not in target_tokens]   
    
    pad_idx = task.source_dictionary.pad()
    
    
    outputs = [[seed_text, 0.0, '']]  
    
    word_num = 0
    
    while word_num < length:
       
       all_candidates = []
       input_sequences = []
       
       
       # add more than one candidate to choose from
       for output in outputs:       # different candidates
           
           generated_text, score, generated_words = output
           # convert the generated text into a list
           sequences = generated_text.split(' ')
           
           for sequence in sequences:    
               # Convert from string to list of units
               sentence_tokens = task.source_dictionary.encode_line(sequence, append_eos=False, add_if_not_exist=False).long()
               # Always generate similar tokens!! but similar tokens do not mean similar words 
               input_sequences.append(sentence_tokens)
           
           sequences_inputs = torch.nn.utils.rnn.pad_sequence(input_sequences, batch_first=True, padding_value=pad_idx).t()
           
           if gpu:
               sequences_inputs = sequences_inputs.cuda(0)
               
           # Compute output & probabilities; shape of the last layer: (sequence_length, batch_size, vocabulary_size)
           
           with torch.no_grad():
               output_ts, _ = generate_model.forward(sequences_inputs)
               
               # add the temperature para    
               scaled_logits = output_ts / temp     
               
           if gpu:
               softmax_output = torch.nn.functional.softmax(scaled_logits,dim=-1).cuda()   
           else:
               softmax_output = torch.nn.functional.softmax(scaled_logits,dim=-1)
           
           for filtered_index in filtered_indices:
               softmax_output[:, :, filtered_index] = -1
                   
           
           sorted_values, sorted_indices = torch.topk(softmax_output, k=beam_width, dim=-1)
               
           # run on gpus when available; 
           if gpu:
                   
               sorted_values = sorted_values.cuda()
               sorted_indices = sorted_indices.cuda()
                   
           indices = sorted_indices.tolist()[-1][-1]   
           word_probs = sorted_values.tolist()[-1][-1]   
           
           n = 0
           
           while n < len(indices):
               
               generated_text_temp = generated_text
               generated_words_temp = generated_words
               index = indices[n]
               
               word = ' ' + token_dict[index]  
               generated_text_temp += word
               generated_words_temp += word
               # record the accumulated probability scores
               new_score = score + np.log(word_probs[n])
                   
               all_candidates.append([generated_text_temp, new_score, generated_words_temp])
                 
               n += 1
       
        
       ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
       
       
       outputs = ordered[:beam_width]
       
       top_candidate = outputs[0][2].split(' ')
       word_num = count_boundary(boundary_lst,top_candidate[2:])   # note here we count the generated words; remove the beginning intials
       
       
    # get the most probable candidate and output the generated tokens(seed text NOT included)
    generated_all = outputs[0][2].replace(' ', '')
   
    return generated_all




def generate_sampling(seed_text,sample_mode,temp,topk,length,token_dict,boundary_lst,target_tokens,task,generate_model,gpu):
    
    '''
    top-k/top-p/random sampling
    
    input: one string sequence
    output: generated string
    '''
    
    
    if gpu:
        generate_model = generate_model.cuda()
    
    
    pad_idx = task.source_dictionary.pad()
    
    generated_words = ''
    generated_text = seed_text
    filtered_indices = [key for key, value in token_dict.items() if value not in target_tokens]   
    
    i = 0
    initial_blank = True
    while i < length:
           
        sequences = generated_text.split(' ')
            
        input_sequences = []
        for sequence in sequences:    
                # encode each char in the word dictionary
            sentence_tokens = task.source_dictionary.encode_line(sequence, append_eos=False, add_if_not_exist=False).long()
            
            # Always generate similar tokens!! but similar tokens do not mean similar words 
            input_sequences.append(sentence_tokens)
            
        if gpu:
            sequences_inputs = torch.nn.utils.rnn.pad_sequence(input_sequences, batch_first=True, padding_value=pad_idx).t().cuda()
        else:
            sequences_inputs = torch.nn.utils.rnn.pad_sequence(input_sequences, batch_first=True, padding_value=pad_idx).t()
        
          
        with torch.no_grad():
            output_ts, _ = generate_model.forward(sequences_inputs)
            
            # add the temperature para    
            scaled_logits = output_ts / temp     
        
        # filter out the unwanted indices
        for filtered_index in filtered_indices:
            scaled_logits[:, :, filtered_index] = float('-inf')
        
        
        # select different sampling methods
        if sample_mode.split('_')[-1] == 'topk':
        
            index = top_k(scaled_logits,topk,gpu)
            
        elif sample_mode.split('_')[-1] == 'topp':
        
            index = top_p(scaled_logits,topk,gpu)
            
        elif sample_mode.split('_')[-1] == 'random':
        
            index = random_sampling(scaled_logits,gpu)
            
        # convert index into token
        word = token_dict[index]      
        generated_words += ' ' + word
        generated_text += ' ' +  word
        
                
        if word not in boundary_lst:
            initial_blank = False
            
        if word in boundary_lst:    
        
            if initial_blank:      
                pass
            
            else:
                i+=1 
        
    generated_words = generated_words.replace(" ", "")        
    return generated_words





def main(argv):
    
    
    # Args parser
    args = parseArgs(argv)
    
    # load eval models and the model used for generation
    gpu = args.gpu 
    
    # load evaluation model
    print("Loading GPT-2 model for evaluation")
    model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large', etc.
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if gpu:    
        eval_model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
    else:
        eval_model = GPT2LMHeadModel.from_pretrained(model_name)
    print("Finished GPT-2 model loading")   
    
    month = args.month
    # path to the training set(input) 
    DictPath = args.DictPath + '/' + month + '/bin_data/dict.txt' 
    ModelPath = args.ModelPath + '/' + month + '/checkpoints' 
    pathLSTMCheckpoint = ModelPath + '/checkpoint_best.pt'
    
    
    # Load dictionary and model checkppoint for inference
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
    
   
    # this is for model path
    DataPath = args.DataPath
    
    OutputPath_temp = args.OutputPath
    mode = args.mode
    prompt_mode = args.prompt_mode
    
    # convert the string to list
    if mode.split('_')[-1]== 'topp':
        sample_lst = [float(x) for x in args.sample_lst.split(',')]  
        
    else:
        sample_lst = [int(x) for x in args.sample_lst.split(',')]  
        
    temp_lst = [float(x) for x in args.temp_lst.split(',')]
    
    boundary_lst = args.boundary_lst
    # filter the unwanted token dictionary
    target_tokens = args.target_tokens
    OutputPath = OutputPath_temp + '/' + month + '/' + prompt_mode +'/'  + mode
    # create a directory if there's no such path
    if not os.path.exists(OutputPath):
        os.makedirs(OutputPath)
    
    # generate tokens and calculate the entropy based on gpt-2
    print("")
    print("Loading prompt data from {}...".format(DataPath))
    
    prompt = pd.read_csv(DataPath).head(2)
    
    # loop over decoding parameters and temperatures
    for decoding_para in tqdm(sample_lst):
        for temp in tqdm(temp_lst):
                
            print('Generating prompts with {}: {} and with temperature: {}'.format(mode,decoding_para,temp))
                # get the generated new frame
            generated_lst = []
            h_lst = []
            segmented_lst = []
            prob_lst = []
            n = 0
            while n < prompt.shape[0]:
                    
                length = len(word2char(prompt['text_cleaned'].tolist()[n]).split('|'))
                    
                if prompt_mode == 'prompted':
                    prompt_text = word2char(prompt['prompt_cleaned'].tolist()[n])
                else:
                    prompt_text = '|'
                        
                try:
                    
                    # select different generation modes
                    
                    if mode.split('_')[0] == 'sample':
                        
                        generated =  generate_sampling(prompt_text,mode,temp,decoding_para,length,token_dict,boundary_lst,target_tokens,task,generate_model,gpu)      
                    
                    elif mode.split('_')[0] == 'beam':
                        generated =  generate_beam(prompt_text,decoding_para,temp,length,token_dict,boundary_lst,target_tokens,task,generate_model,gpu)
                        
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
                    h, prob = calculate_entropy(eval_model,tokenizer,segmented,gpu)
                        
                except:
                    generated = 'NA'
                    h = 'NA'
                    prob = 'NA'
                    segmented = 'NA'
                        
                    print('Something wrong with {}: {} and with temperature: {}'.format(mode,decoding_para,temp))
                    print(prompt['text_cleaned'].tolist()[n])
                       
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
                
            prompt_cleaned = prompt[(prompt['LSTM_generated' ] != 'NA')]
                
            # save the file in file inthe naming convention: topk_temp_generated.csv
            prompt_cleaned.to_csv(OutputPath + '/' + str(decoding_para) + '_' + str(temp) + '_generated.csv',escapechar='?')     
            print('Finished generation with {}: {} and with temperature: {}'.format(mode,decoding_para,temp))
                
    print("Finished generation!")

   
    
    
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    



