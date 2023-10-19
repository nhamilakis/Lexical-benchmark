#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
util func for sentence generation
search algorithms for sentence generation
"""
import numpy as np
import re
import json
import argparse
import torch
from fairseq import tasks, checkpoint_utils
import multiprocessing as mp


'''
import sentence processing func
'''

def generate_one(args):
    output_path, word_dict, seed_text, max_length, beam_width, pad_idx, task, model = args
            
    generated_text = seed_text
    generated_words = ''
    outputs = [[seed_text, 0.0]]
    # tokenize the seed text
    tokenized_input = task.source_dictionary.encode_line(generated_text, append_eos=False, add_if_not_exist=False)
    
    for i in range(max_length):
        all_candidates = []
        input_sequences = []
        
        for output in outputs:
            generated_text, score = output
            sequences = generated_text.split(' ')
            
            for sequence in sequences:
                sentence_tokens = task.source_dictionary.encode_line(sequence, append_eos=False, add_if_not_exist=False).long().to('cuda:0')
                input_sequences.append(sentence_tokens)
            
            sequences_inputs = torch.nn.utils.rnn.pad_sequence(input_sequences, batch_first=True, padding_value=pad_idx).t().to('cuda:0')
            output_ts, _ = model(sequences_inputs)
            sorted_values, sorted_indices = torch.topk(output_ts, k=beam_width, dim=-1)
            sorted_values = sorted_values.to('cuda:0')
            sorted_indices = sorted_indices.to('cuda:0')
            indices = sorted_indices.tolist()[-1][-1]
            word_probs = sorted_values.tolist()[-1][-1]
            
            n = 0
            while n < len(indices):
                index = indices[n]
                try:
                    word = ' ' + list(word_dict)[index]
                    generated_words += word
                    generated_text += word
                    new_score = score + np.log(word_probs[n])
                    all_candidates.append([generated_text, new_score])
                except:
                    pass

        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        outputs = ordered[:beam_width]
                
    return [seed_text, generated_words, tokenized_input.tolist(), outputs[0][0]]

def generate(model, task, output_path, word_dict, seed_texts, max_length, num_processes):
    beam_width = 3
    pad_idx = task.source_dictionary.pad()
    
    args_list = [(output_path, word_dict, seed_text, max_length, beam_width, pad_idx, task, model) for seed_text in seed_texts]
    
    with mp.Pool(num_processes) as pool:
        results = pool.map(generate_one, args_list)
        
    return results


def readArgs(pathArgs):
    print(f"Loading args from {pathArgs}")
    with open(pathArgs, 'r') as file:
        args = argparse.Namespace(**json.load(file))
    return args

def writeArgs(pathArgs, args):
    print(f"Writing args to {pathArgs}")
    with open(pathArgs, 'w') as file:
        json.dump(vars(args), file, indent=2)




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


# Define the BPE tokenizer
def bpe_tokenize(text,dictionary):
    # Tokenize the text into words or subwords
    tokens = re.findall(r'\w+|[^\w\s]', text)
    
    encoded_tokens = []
    for token in tokens:
        subwords = list(token)
        while len(subwords) > 1:
            # Find the most frequent pair of subwords
            max_freq = -1
            max_pair = None
            for i in range(len(subwords)-1):
                pair = (subwords[i], subwords[i+1])
                if pair in dictionary and dictionary[pair] > max_freq:
                    max_freq = dictionary[pair]
                    max_pair = pair
    
            if max_pair is None:
                break
    
            # Merge the most frequent pair of subwords
            new_subwords = []
            i = 0
            while i < len(subwords):
                if i < len(subwords)-1 and (subwords[i], subwords[i+1]) == max_pair:
                    new_subwords.append(max_pair)
                    i += 2
                else:
                    new_subwords.append(subwords[i])
                    i += 1
            subwords = new_subwords
        # put the subwords togther
        updated = ''.join(subwords)
        # Convert subwords to IDs
        try:
            index = dictionary[updated] 
        except:
            # !!! possibly for future change as we need to convert it to unknown tokens
            index = dictionary['##a']
        encoded_tokens.append(index)
    return encoded_tokens

def read_file(pathQuantizedUnits,prompt_len):
    input_file_seqs = []
    with open(pathQuantizedUnits, 'r') as f:
        for line in f:
            non_empty_lines = [line for line in line.split("\n") if line.strip() != ""]
            cleaned = "\n".join(non_empty_lines)
            
            if len(cleaned) > 3:
                # get the length of each string
                string_lst_temp = cleaned.split(' ')
                # remove additional space in the list
                string_lst = [x for x in string_lst_temp if x.strip()]
                if len(string_lst) > prompt_len -1:
                    input_file_seqs.append(" ".join(string_lst[:prompt_len]))   
    return input_file_seqs




def prompt_freq(text,dictionary):
    '''
    return a list of the ngram freq
    
    '''
    return None
'''
Beam search
evaluating each possible sequence of words and selecting the one with the highest probability. 
Tstarts by selecting the top K most likely words and then generates all possible sequences of length K+1 that can be formed 
by adding a new word to each of the K sequences. The K+1 sequences are then ranked according to their probability, and the top K are selected to 
generate the next set of possible sequences.
'''
def beam_search(model, seed_text, max_length, beam_width):
    sequences = [[seed_text, 0.0]]
    while len(sequences[0][0]) < max_length:
        all_candidates = []
        for seq in sequences:
            text, score = seq
            next_words = model.predict(text)[-1]
            top_words = next_words.argsort()[-beam_width:]
            for word in top_words:
                new_text = text + ' ' + model.index_word[word]
                new_score = score + np.log(next_words[word])
                all_candidates.append([new_text, new_score])
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:beam_width]
    return sequences[0][0]


'''
randomly selecting the next word based on the probability distribution of the language model. The probability of each word is calculated 
by the language model based on the context of the previous words. Sampling allows for more diverse sentence generation since it does not always 
select the most likely word.
'''


def sample_text(model, seed_text, max_length, temperature):
    generated_text = seed_text
    for i in range(max_length):
        next_word_probs = model.predict(generated_text)[-1]
        next_word_probs = np.asarray(next_word_probs).astype('float64')
        next_word_probs = np.log(next_word_probs) / temperature
        next_word_probs = np.exp(next_word_probs) / np.sum(np.exp(next_word_probs))
        index = np.random.choice(len(next_word_probs), p=next_word_probs)
        next_word = model.index_word[index]
        generated_text += ' ' + next_word
    return generated_text


def top_k_sampling(model, seed_text, max_length, k):
    generated_text = seed_text
    for i in range(max_length):
        next_word_probs = model.predict(generated_text)[-1]
        top_k_words = next_word_probs.argsort()[-k:]
        top_k_probs = next_word_probs[top_k_words] / sum(next_word_probs[top_k_words])
        index = np.random.choice(top_k_words, p=top_k_probs)
        next_word = model.index_word[index]
        generated_text += ' ' + next_word
    return generated_text


'''
Temperature Scaling: This algorithm is a variation of sampling that allows for more control over the randomness of the generated sentences. It scales the probabilities of the words by a temperature parameter that controls how much the probabilities are randomized. Lower temperature values result in more conservative, predictable sentences, while higher temperature values lead to more random, unpredictable sentences.

'''


def temperature_sampling(model, seed_text, max_length, temperature):
    generated_text = seed_text
    for i in range(max_length):
        next_word_probs = model.predict(generated_text)[-1]
        next_word_probs = np.asarray(next_word_probs).astype('float64')
        next_word_probs = np.log(next_word_probs) / temperature
        next_word_probs = np.exp(next_word_probs) / np.sum(np.exp(next_word_probs))
        index = np.random.choice(len(next_word_probs), p=next_word_probs)
        next_word = model.index_word[index]
        generated_text += ' ' + next_word
    return generated_text



'''
Greedy Decoding: This algorithm generates the sentence word by word, always selecting the most likely word according to the language model's probability distribution. This approach can produce coherent sentences but can also lead to repetition and lack of diversity in the generated sentences.

'''

def greedy_decoding(model, seed_text, max_length):
    generated_text = seed_text
    for i in range(max_length):
        next_word_probs = model.predict(generated_text)[-1]
        index = np.argmax(next_word_probs)
        next_word = model.index_word[index]
        generated_text += ' ' + next_word
    return generated_text



