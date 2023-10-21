#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize the generation parameters from the LSTM reference model

input: .csv dataframe with parents and children's utterances, segmented into turns
ouput: .csv with the generated tokens as an additional column

@author: jliu
"""
import pandas as pd
import torch
from generation_util import loadLSTMLMCheckpoint,calculate_fitness
import os 
import numpy as np
import sys
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from skopt.space import Real, Integer
from skopt import gp_minimize
from skopt.plots import plot_objective,plot_evaluations
import random


'''
load generation model
'''
# some arbituatury para
month = '36'
DictPath = '/scratch2/jliu/STELAWord/productive_vocab/reference/data/' + month + '/bin_data/dict.txt'
ModelPath = '/scratch2/jliu/STELAWord/productive_vocab/reference/model/'+ month +'/checkpoints'
pathLSTMCheckpoint = ModelPath + '/checkpoint_best.pt'
# this is for model path
DataPath = '/scratch2/jliu/STELAWord/productive_vocab/reference/eval/Prompt_AE.csv'
OutputPath = '/scratch2/jliu/STELAWord/productive_vocab/reference/eval/'+ month



DictPath = 'bin_data/dict.txt'
ModelPath = 'checkpoints'
pathLSTMCheckpoint = ModelPath + '/checkpoint_best.pt'
# this is for model path
DataPath = 'Prompt_AE.csv'
OutputPath = 'generated'

# load the prompt data
prompt = pd.read_csv(DataPath)[:10]
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

generate_model, task = loadLSTMLMCheckpoint(pathLSTMCheckpoint, ModelPath)
generate_model.eval()
print("Model loaded !")




'''
load evaluation model
'''
model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large', etc.
eval_model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

   

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Generate tokens from decoder models')
    
    parser.add_argument('--gpu', type=bool, default = False,
                        help='Whether to use GPU.')
    
    
    return parser.parse_args(argv)

 
def generate_beam(seed_text,beam_width,temp,topk,length,word_dict,task,generate_model):
    
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
            
            
            sorted_values, sorted_indices = torch.topk(softmax_output, k=topk, dim=-1)
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
    generate tokens randomly
    
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

    return generated_tokens
    



def calculate_entropy(eval_model,sentence):
    
    
    '''
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



def objective_function(param_list):
    
    '''
    perform model generation with the target hyper
    take the prompt frame as the gloabl variable
    '''
    beam_width = param_list[0]
    temp = param_list[1]
    topk = param_list[2]
    # get the generated new frame
    generated_lst = []
    h_lst = []
    n = 0
    while n < prompt.shape[0]:
        
        # keep the length of generated tokens simialr as ground truth
        length = len(prompt['text_cleaned'].tolist()[n].split())
        
        try:
            # generate the prompt
            generated = generate_beam(prompt['prompt_cleaned'].tolist()[n],beam_width,temp,topk,length,word_dict,task,generate_model)
            #calculate the entropy
            h = calculate_entropy(eval_model,generated)
        except:
            generated = 'NA'
            h = 'NA'
        
        generated_lst.append(generated)
        h_lst.append(h)
        n += 1
    # only preserve the selected data and the results! 
    prompt['reference_generated'] = generated_lst
    prompt['reference_generated_h'] = h_lst
    
    # remove the error rows
    prompt_cleaned = prompt[(prompt['reference_generated'] != 'NA') & (prompt['reference_generated_h'] != 'NA')]
    
    '''
    calculate the fitness between children's utterances and model's generation  
   '''
    mean_values = prompt_cleaned.groupby('month')['reference_generated_h'].mean()
    mean_value_CHILD = prompt_cleaned.groupby('month')['text_cleaned_h'].mean()
    rmse = calculate_fitness(mean_values, mean_value_CHILD)
    # save the generated results with the rmse score as the file name
    prompt_cleaned.to_csv(OutputPath + '/' + str(rmse) + '_' + str(topk)+'_'+str(beam_width)+'_'+ str(temp) + '.csv')
    return rmse



    
    

def main(argv):
    # Args parser
    args = parseArgs(argv)
    
   
    # create a directory if there's no such path
    if not os.path.exists(OutputPath):
        os.makedirs(OutputPath)
        
   
    print("Finished generation!")
    
    # Define the search space
    '''
    space = [
        Integer(1, 6, name='beam_width'), # range for beam_width
        Real(0.1, 1, name='temp'),  # range for temperature
        Integer(2, 4, name='topk')    # range for top-k

    ]
    '''
    
    print("Optimizing hyperparameters")
    space = [
        Integer(1, 3, name='beam_width'), # range for beam_width
        Integer(1, 3, name='temp'),  # range for temperature
        Integer(2, 4, name='topk')    # range for top-k

    ]
    # preserve the results as the csv file
    result = gp_minimize(objective_function, space, n_calls=20, random_state=0)
    best_beam_width, best_temp, best_topk = result.x
    best_objective_value = result.fun
    
    para_list = [best_beam_width, best_temp,best_topk,best_objective_value]
    
    para_frame = pd.DataFrame([para_list])
                 
    para_frame = para_frame.rename(columns={0: 'Best beam width', 1: 'Best Temperature',2:'Best Top-k',3:'Best Objective Value'})
               
    para_frame.to_csv(OutputPath + 'Optimal_para.csv')
    
    print(f"Best beam width: {best_beam_width}")
    print(f"Best Temperature: {best_temp}")
    print(f"Best Top-k: {best_topk}")
    print(f"Best Objective Value: {best_objective_value}")
    
    # save in a csv file 
    
    
    # plot_objective(result)
    # plot_evaluations(result)  
        
   
    
    
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    



