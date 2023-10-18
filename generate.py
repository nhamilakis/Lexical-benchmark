#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize the generation parameters from the LSTM reference model

input: .csv dataframe with parents and children's utterances, segmented into turns
ouput: .csv with the generated tokens as an additional column

TO DO: 1. integrate gpu settings
    2. note this is still the top-down search of the optimal para, we stil need to inporate an objective func to optimize theese two 
        OR
      integrate the Bayesian search for hyperpara optimization: first get the whole pipeline


@author: jliu
"""
import pandas as pd
import torch
from generation_util import loadLSTMLMCheckpoint
import os 
import numpy as np
import sys
import argparse

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Generate tokens from decoder models')
    parser.add_argument('--ModelPath', type=str, default = 'checkpoints',
                        help='Path to the pretrained language model; even choose whether to use best checkpoint or not')
    
    parser.add_argument('--DictPath', type=str, default = 'bin_data/dict.txt',
                        help='Path to the dictionary data')
    
    parser.add_argument('--DataPath', type=str, default = 'Prompt_AE.csv',
                        help='Path to the parent-child utterance file')
    
    parser.add_argument('--OutputPath', type=str, default = 'generated',
                        help='Path to the generation output.')
    
    parser.add_argument('--para', type=str, default = 'top-k',
                        help='Tested hyperparameters, either top-k or temp')
    
    parser.add_argument('--k_list', type=bool, default = [3],
                        help='a list of top-k candidates for optimization')
    
    parser.add_argument('--temp_list', type=bool, default = [0.01,1,10],
                        help='a list of temperature parameters for optimization')
    
    
    parser.add_argument('--gpu', type=bool, default = True,
                        help='Whether to use GPU.')
    
    parser.add_argument('--batch_size', type=int, default = 128,
                        help='batch_size for prallell  processing.')
    
   
    return parser.parse_args(argv)





'''
add temperature parameter to the model generation
'''  


  
def generate(seed_text,beam_width,temp,length,word_dict,task,model):
    
    '''
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
                output_ts, _ = model(sequences_inputs)
                
                # add the temperature para    
                scaled_logits = output_ts / temp     #!!! something wrong here
                
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
            
            # use word probability to 
            word_probs = sorted_values.tolist()[-1][-1]     
            
            # prepare for the recursion 
            n = 0
            
            # note that there are multiple generated indices
            while n < len(indices):
                generated_words = ''
                generated_text_temp = generated_text
                index = indices[n]
                # expand different word sequences in this condition; to be updated on this level
                try:
                    
                    '''
                    note this should be re-started every time
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
    
    


'''
perform Bayesian search on hyperparameters
'''


def main(argv):
    # Args parser
    args = parseArgs(argv)
    
    
    # path to the training set(input) 
    DictPath = args.DictPath
   
    ModelPath = args.ModelPath
    pathLSTMCheckpoint = ModelPath + '/checkpoint_best.pt'
    # this is for model path
    DataPath = args.DataPath
    k_list = args.k_list
    temp_list = args.temp_list
    OutputPath = args.OutputPath
    
    # Load input file
    print(f"Reading dictionary from {DictPath}")
    word_dict = {}
    with open(DictPath, 'r') as f:
        for line in f:
        
            non_empty_lines = [line for line in line.split("\n") if line.strip() != ""]
            cleaned = "\n".join(non_empty_lines)
            word_dict[cleaned.split(' ')[0]] = int(cleaned.split(' ')[1])
    print("Dictionary loaded!")
    
    
    # Load LSTM model
    print("")
    print(f"Loading LSTM model from {pathLSTMCheckpoint}...")
    
    model, task = loadLSTMLMCheckpoint(pathLSTMCheckpoint, ModelPath)
    model.eval()
    print("Model loaded !")
    

    # Run and save outputs
    print("")
    print("Loading prompt data from {pathQuantizedUnits}...")
    prompt = pd.read_csv(DataPath).head(5)
    
    
    
    # experiment on different beam sizes
    print("Generating tokens from the prompts and saving results...")
    for k in k_list:
        
        for temp in temp_list:
         # Run and save outputs
            generated_lst = []
           
            n = 0
            while n < prompt.shape[0]:
                
                # keep the length of generated tokens simialr as ground truth
                length = len(prompt['text_cleaned'].tolist()[n].split())
                
                try:
                    generated = generate(prompt['prompt_cleaned'].tolist()[n],k,temp,length,word_dict,task,model)
                    
                    
                except:
                    generated = 'NA'
                    
                generated_lst.append(generated)
                n += 1
        
            prompt['reference_generated_' + str(k)+'_'+str(temp)] = generated_lst
        
    # create a directory if there's no such path
    if not os.path.exists(OutputPath):
        os.makedirs(OutputPath)
        
    prompt.to_csv(OutputPath + '/generated.csv')
    print("Finished generation!")

   
    
    
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    



