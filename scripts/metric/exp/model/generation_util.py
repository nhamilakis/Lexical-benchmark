#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
util func for sentence generation
search algorithms for sentence generation
"""

import re
import argparse
from fairseq import tasks, checkpoint_utils
import torch
import subprocess
import string

'''
import sentence processing func
'''


def run_command(command):
    subprocess.call(command, shell=True)
    

def random_sampling(scaled_logits,gpu):
    
    if gpu:
        softmax_output = torch.nn.functional.softmax(scaled_logits, dim=-1).cuda() 
        
        # Perform random sampling among the top-p tokens
        index = torch.multinomial(softmax_output[0][-1], num_samples=1).cuda() 
    
    else:
        
        softmax_output = torch.nn.functional.softmax(scaled_logits, dim=-1)
        
        # Perform random sampling among the top-p tokens
        index = torch.multinomial(softmax_output[0][-1], num_samples=1)
        
    return index.item()


def top_k(scaled_logits,topk,gpu):
    # select top-k candidates 
    sorted_values, sorted_indices = torch.topk(scaled_logits, k=topk,dim=-1)
        
    if gpu:
        
        # run on gpus when available
        sorted_values = sorted_values.cuda()
        sorted_indices = sorted_indices.cuda()
        
        # apply softmax on the selected results; this is already normalized
        softmax_output = torch.nn.functional.softmax(sorted_values,dim=-1).cuda() 
        
        # Randomly sample an index based on the probability distribution of the predicted next word
        sampled_index = torch.multinomial(softmax_output[0][-1], num_samples=1).cuda()   # Perform sampling
    
        
    else:
       
        # apply softmax on the vocabulary
        softmax_output = torch.nn.functional.softmax(sorted_values,dim=-1)
        
        # Randomly sample an index based on the probability distribution
        sampled_index = torch.multinomial(softmax_output[0][-1], num_samples=1)
        
    index = sorted_indices.tolist()[-1][-1][sampled_index.item()] 
    return index    




def top_p(scaled_logits,p,gpu):
    
    
    if gpu:
        
        softmax_output = torch.nn.functional.softmax(scaled_logits, dim=-1).cuda()
        
    else:
        softmax_output = torch.nn.functional.softmax(scaled_logits, dim=-1)    
        
    
        
    # Sort the probabilities and indices in descending order
    sorted_probs, sorted_indices = torch.sort(softmax_output, descending=True, dim=-1)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Getthe smallest set of tokens whose cumulative probability exceeds threshold p
    cutoff_index = torch.where(cumulative_probs > p)[-1][0] + 1
    
    top_p_probs = sorted_probs[:, :, :cutoff_index]
    
    rescaled_probabilities = top_p_probs / top_p_probs.sum(dim=-1, keepdim=True)
    
    
    if gpu:  
        
        # Perform random sampling among the top-p tokens
        sampled_index = torch.multinomial(rescaled_probabilities[0][-1], num_samples=1).cuda()
    
    
    else:
        
        # Perform random sampling among the top-p tokens
        sampled_index = torch.multinomial(rescaled_probabilities[0][-1], num_samples=1)
    
    
    index = sorted_indices.tolist()[-1][-1][sampled_index.item()]
    
    return index








def calculate_entropy(eval_model,tokenizer,sentence,gpu):

    """
    input: the single text string
    output: the calculated entropy score
    """
    
    # Tokenize the sentence and convert to tensor
    if gpu:
        input_ids = tokenizer.encode(sentence, return_tensors="pt").cuda()
    else:
        input_ids = tokenizer.encode(sentence, return_tensors="pt")
    
    
    # Get the logits from the model
    with torch.no_grad():
            logits = eval_model(input_ids).logits[0]
    
    # probability distribution of all the tokens in the vocabulary
    if gpu:
        probabilities = torch.nn.functional.softmax(logits, dim=-1).cuda()
    else:
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
    # Calculate entropy    thisis fine as we are getting all tokens in the vocab size
    if gpu:
        entropy = -torch.sum(probabilities * torch.log2(probabilities)).cuda()
    else:
        entropy = -torch.sum(probabilities * torch.log2(probabilities))
    # normalize by the sequence length; otherwise this would be only the reflection of length distr
    norm_entropy = entropy/probabilities.shape[0] 
    
   
    # get the average log prob of the given sequence !loop the index list 
    i = 0
    
    log_prob = 0
    for idx in input_ids.tolist()[0]:
        try:
            log_prob += torch.log2(probabilities[i][idx]).item()
        except:     
            log_prob += 0
        i += 1
        
    avg_log_prob = log_prob/i * (-1)    
    
    return norm_entropy.item(),avg_log_prob



def count_boundary(boundary_lst, generated_lst):
    
    '''
    count the number of word boundaries in a text string
    '''
    count_dict = {}
    for element in boundary_lst:
        count_dict[element] = generated_lst.count(element)
    return sum(count_dict.values())



def token_index(task,word_dict):
    # add an additional word dictionary token list
    token_lst = list(word_dict.keys())
    token_lst.extend(['<pad>', '<unk>', '<s>', '</s>'])
    token_dict = {}
    for token in token_lst:
        index = task.source_dictionary.encode_line(token, append_eos=False, add_if_not_exist=False).long().tolist()[0]
        token_dict[index] = token
    return token_dict   





def loadLSTMLMCheckpoint(pathLSTMCheckpoint, pathData):
    """
    Load lstm_lm model from checkpoint.
    """
    # Set up the args Namespace
    model_args = argparse.Namespace(
        task='language_modeling',
        output_dictionary_size=-1,
        data=pathData,
        path=pathLSTMCheckpoint
        )

    # Setup task
    task = tasks.setup_task(model_args)
    
    # Load model
    models, _model_args = checkpoint_utils.load_model_ensemble([model_args.path], task=task)
    model = models[0]
    
    return model, task



def word2char(text):
    
    '''
    convert word seq into char se with word boundary markers as |
    return: formated strings and the length of the string
    
    '''
    translator = str.maketrans('','',string.punctuation + string.digits)
    
    temp= text.translate(translator)
    clean = ' '.join(temp.split())
    #  remove the beginning and ending space
    cleaned = clean.strip(' ')    
    
    
    temp= cleaned.upper() 
    
    # replace blank with "|"
    clean_temp = temp.replace(" ", "|")
    # remove redundent consecutive "|"
    clean = re.sub(r'\|{2,}', '|', clean_temp)
    #  remove the beginning and ending '|'
    cleaned = clean.strip('|')                
    # add space after each char
    cleaned = " ".join(cleaned)
    return cleaned         












