# -*- coding: utf-8 -*-
"""

@author: Jing Liu

finetune hyperparameters of genration

input: reference LSTM model checkpoints (really wanna do greedy search by integrate these two factors into the losss function)
output: 1. figures of entropy distribution in para different settings; 2 .csv file with the the fitness scores
    
"""
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
import os
import argparse
import sys
from tqdm import tqdm
from time import time

'''
model configurations and settings
'''

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Generate tokens from decoder models')
    parser.add_argument('--ModelPath', type=str, default = '/scratch2/mrita/opt_jing/checkpoint-26410/',
                        help='Path to the pretrained language model')
    
    parser.add_argument('--DataPath', type=str, default = '/scratch2/jliu/BabyLM/babylm-train/babylm_data/babylm_10M',
                        help='Path to the concatenated train file.')
    
    parser.add_argument('--OutputPath', type=str, default = '/scratch2/jliu/BabyLM/babylm-train/babylm_data/babylm_generated_temp',
                        help='Path to the generation output.')
    
    
    parser.add_argument('--Promptlen', type=int, default = 5,
                        help='Path to the generation output.')
    
    parser.add_argument('--gpu', type=bool, default = True,
                        help='Whether to use GPU.')
    
    parser.add_argument('--batch_size', type=int, default = 128,
                        help='batch_size for prallell  processing.')
    
    
    return parser.parse_args(argv)


    
def load_data(DataPath, prompt_len, start_prop, end_prop):
    
    '''
    input: the file path containing all the extracted prompts
    output:a frame with selected prompts
    '''
    
    # check whether to concatenate train data
    if os.path.exists(DataPath + '/All_train.csv'):
        print('  ------Data have been concatenated! --------    ')
        
    else: 
        print('  ------Concatenating data--------    ')
        
        final_frame = pd.DataFrame()
        for file in tqdm(os.listdir(DataPath)): 
            file_name = os.fsdecode(file)
            
            if file_name.endswith('.train'):
                
                with open(DataPath + '/' + file_name, 'r', encoding = 'utf-8') as new_file:
                    text = new_file.readlines()

                    # count the number of words and punct
                    number_lst = []
                    for word in text:
                        number = word.count(' ') + 1
                        number_lst.append(number)
                        
                    text_frame = pd.DataFrame([text,number_lst]).T
                    text_frame = text_frame.rename(columns={0: 'text', 1: 'length'})
                    text_frame['filename'] = file_name
                    final_frame = pd.concat([final_frame,text_frame])
        
        final_frame.to_csv(DataPath + '/All_train.csv')
    
    print('  ------Selecting first {} words--------    '.format(prompt_len))
    
    # select the prompts
    all_sent = pd.read_csv(DataPath + '/All_train.csv')
    selected_frame = all_sent[all_sent['length'] > (int(prompt_len) - 1)]
    
    selected_prompt_lst = []
    for selected_text in tqdm(selected_frame['text'].tolist()):
        # get the prompt
        selected_prompt = ' '.join(selected_text.split(' ')[:prompt_len])
        selected_prompt_lst.append(selected_prompt)
        
    selected_frame['prompt'] = selected_prompt_lst
    
    selected_frame_new = selected_frame.drop_duplicates(subset=['prompt'])
    selected_frame = selected_frame_new.reset_index()
    selected_frame.to_csv(DataPath + '/Prompt_' + str(prompt_len) + '.csv')
    
    filename_lst = set(selected_frame['filename'].tolist())
    cropped_all = pd.DataFrame()
    for file in filename_lst:
        cleaned_selected = selected_frame[selected_frame['filename'] == file]
        cropped_selected = cleaned_selected[round(start_prop*cleaned_selected.shape[0]) : round(end_prop*cleaned_selected.shape[0])]
        cropped_all = pd.concat([cropped_all,cropped_selected])
    
    
    print('  ------Finished words selection! ------    ')
    
    
    return cropped_all






def generate_token(tokenizer,model,prompt,temp,gpu): 
    
    '''
    input: prompt -> string
    output:  generated seq, ppl
    '''
    
    inputs = tokenizer([prompt], return_tensors="pt") 
    if gpu:
        inputs = inputs.to('cuda')
    
    # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for encoder-decoder models, like BART,OPT or T5.
    
    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
        
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        num_beams=5,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True,
        temperature = temp
    )
    
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
    )
    
    generated_tokens = outputs.sequences[:, input_length:]
    
    transition_scores_cpu = transition_scores.cpu().numpy()
    output_length = input_length + np.sum(transition_scores_cpu < 0, axis=1)
    # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
    # Tip: recomputing the scores is only guaranteed to match with `normalize_logits=False`. Depending on the use case, you might want to recompute it with `normalize_logits=True`.
   
    length_penalty = model.generation_config.length_penalty
    reconstructed_scores = transition_scores_cpu.sum(axis=1) / (output_length**length_penalty)

    # calculate perplexity
    
    ppl = math.exp((-1) * sum(np.log2(np.exp(reconstructed_scores)).tolist())/len(np.log2(np.exp(reconstructed_scores)).tolist()))
    
    
    token_lst = []
    for tok in generated_tokens[0]:
       token_lst.append(tokenizer.decode(tok))
    # convert the list into string
    tokens = ' '.join(token_lst)
    
    return tokens, ppl



def generate_token_batch(selected_frame,tokenizer,model,beam_num,gpu): 
    
    '''
    input: subframe of each batch
    output: updated subframe with generated seq, ppl as 2 aditional columns
    
    '''
    token_lst = []
    ppl_lst = []
    for prompt in tqdm(selected_frame['prompt'].tolist()):
    
        tokens, ppl = generate_token(tokenizer,model,prompt,beam_num,gpu)
        
        token_lst.append(tokens)
        ppl_lst.append(ppl)
        
    
    selected_frame['generated_' + str(beam_num)]= token_lst
    selected_frame['ppl_' + str(beam_num)]= ppl_lst
    
    return selected_frame
    
    


def main(argv):
    # Args parser
    args = parseArgs(argv)
    # read data and generate the dataframe with selected prompts
    start_prop = args.start_prop
    end_prop = args.end_prop
    selected_frame = load_data(args.DataPath, args.Promptlen, start_prop, end_prop)
    print("Number of sequences: {}".format(selected_frame.shape[0]))
    
    gpu = args.gpu
    batch_size = args.batch_size
    print('Loading model from {}'.format(args.ModelPath))

    tokenizer = AutoTokenizer.from_pretrained(args.ModelPath)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.ModelPath)
    
    if gpu:
       model = model.to('cuda') 
       
       
    # go through all the different para lists
    temp_lst = [50,5000]
    
    for temp in temp_lst:
        
        print('  ------generating data with temperature {}--------    '.format(str(temp)))
        
        n_batch = selected_frame.shape[0]//batch_size
        if selected_frame.shape[0] % batch_size != 0:
            n_batch += 1
        
        all_frame = pd.DataFrame()
        start_time = time()
        
        n = 0
        for i in tqdm(range(n_batch)):
            selected_frame_batch = selected_frame[i*batch_size : min((i+1)*batch_size, selected_frame.shape[0])]
            generated_frame = generate_token_batch(selected_frame_batch,tokenizer,model,temp,gpu)
            save_path = args.OutputPath + '/' + str(temp) + '/'+ str(start_prop)+ '-'+ str(end_prop)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            generated_frame.to_csv(save_path +'/trial_beam_10M_'+str(n)+'.csv') 
            
            all_frame = pd.concat([all_frame,generated_frame])
            n += 1
            
        all_frame.to_csv(args.OutputPath  + '/generated_beam_10M_'+str(temp) +'.csv') 
        
        print("Done all in {:4f} s.".format(time() - start_time))
        
        print('  ------Finished generating data with temperature {}--------    '.format(str(temp)))
    
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    
    


