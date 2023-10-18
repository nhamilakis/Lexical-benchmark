# -*- coding: utf-8 -*-
"""
Calculate ppl in the chronological order

@author: Jing Liu
"""

import torch
from transformers import pipeline, set_seed
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns   
import os
import enchant
import numpy as np
from util import get_prompt
# load the pretrained LM
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large', etc.
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

d = enchant.Dict("en_US")

def parseArgs(argv):
    # Run parameters: we have similar folder structure, so juat input lang
    parser = argparse.ArgumentParser(description='Investigate CHILDES corpus')
    
    parser.add_argument('--lang', type=str, default = 'BE',
                        help='American or British English to operate on')
    
    parser.add_argument('--generate', type=str, default = False,
                        help='whether to generate new tokens based on the prompt')
    
    
    return parser.parse_args(argv)


TextPath = 'Output/AE/production/ppl/generated/calculated/'
# preprocess the uncleaned data 
generated_all = pd.DataFrame()
for file in os.listdir(TextPath): 
    frame = pd.read_csv(TextPath + file)
    generated_all = pd.concat([generated_all,frame])

generated_all.to_csv(TextPath + 'Calculated.csv')

def load_data(text_path):
    
    '''
    clean the dataframe with the transcript
    input: the dataframe with all the required columns and the cleaned ones
    output: the chosen column as the reference
    '''
    # read the original data
    generated_all = pd.read_csv(text_path +'/Prompt_clean_old.csv')
    
    def clean_data(generated_all, header):
        # remove the annotation line
        words_to_check = ['xxx', 'yyy', 'noise', 'rustling', 'vocalize', 'vocalization','breath','sound','babbl','oh',
                          'ow','cough','um','cry','eh','cries','moo', 'bang','coo','rasberries','transcribe','ha',
                          'sneez','bubbl','squeal','\x15','trn', '.']
        generated_all = generated_all[~generated_all[header].str.isspace()]
        cleaned_lst = []
        for line in generated_all[header].tolist():
            # remove the keywords/annotations recursively
            for check in words_to_check:
                line = line.replace(check,'')
            cleaned_lst.append(line)
        generated_all[header + '_cleaned'] = cleaned_lst
        generated_all = generated_all[~generated_all[header + '_cleaned'].str.isspace()]
        return generated_all
    
    generated_all = clean_data(generated_all, 'text')
    generated_all = clean_data(generated_all, 'prompt')
    generated_all.to_csv(text_path +'/Prompt_clean.csv')
    return generated_all


def get_prob(frame):
    
    '''
    input: the frame and header to calculate 
    output: the concatented text dataframe with
            1. the calculated utterance prob (averaged form)
            2. the calculated utterance entropy 
            for: 1. children's utterances
                 2. parents' utterances
                 3. generated tokens
    '''
    
    def calculate(sentence):
        
        '''
        input: single text string 
        output: calculated entropy and prob
        '''
        # Tokenize the sentence and convert to tensor
        input_ids = tokenizer.encode(sentence, return_tensors="pt")
    
        # Get the logits from the model
        with torch.no_grad():
            logits = model(input_ids).logits[0]
    
        # Calculate the probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
        # Calculate entropy
        entropy = -torch.sum(probabilities * torch.log2(probabilities))
    
        # Calculate the average probability
        average_probability = torch.mean(probabilities)

        return average_probability.item(),entropy.item()

    # remove duplicated rows given the chosen column
    
    header_lst = ['generated', 'prompt_cleaned', 'text_cleaned']
    for header in header_lst:
        prob_lst = []
        h_lst = []
        frame = frame[~frame.duplicated(subset=header, keep=False)]
        
        for text in list(set(frame[header].tolist())):
            # in case that the given sequence cannot be calculated from the model
            try:
                prob,h = calculate(text)
                required_length = frame[frame[header] == text].shape[0]
            except:
                # use place holder to fill in it
                prob = 'NA'
                h = 'NA'
            prob_lst.append([prob] * required_length)
            h_lst.append([h] * required_length)
            
        
        # unfold the sublists
        prob_lst = [item for sublist in prob_lst for item in sublist]
        h_lst = [item for sublist in h_lst for item in sublist]
        # append to the dataframe
        frame[header + '_prob'] = prob_lst
        frame[header + '_h'] = h_lst
     
    return frame
    


def plot_all(generated_all,lang):
    
    def plot_prob(child_trans,title,header):
        
        # remove the duplicated ones
        child_trans = pd.DataFrame([child_trans['month'].tolist(), child_trans[header].tolist()]).T
        child_trans = child_trans.rename(columns={0: "month",1: header})
        sns.set_style('whitegrid') 
        
        if header.startswith('text_cleaned'):
            label = 'child'
        
        elif header.startswith('prompt_cleaned'):
            label = 'parent'
        
        elif header.startswith('generated'):
            label = 'generated'
        
        ax = sns.lineplot(x="month", y=header, data=child_trans, label = label)
        plt.title(title, fontsize=20)
        # set the limits of the x-axis for each line
        for line in ax.lines:
            plt.xlim(0,36)
            plt.ylim(0,0.0001)
        plt.ylabel('probability')
    
    if lang == 'AE':
        title = 'American '
    elif lang == 'BE':
        title = 'British '
    plot_prob(generated_all, title + 'English','text_cleaned_prob')
    plot_prob(generated_all, title + 'English','prompt_cleaned_prob')
    plot_prob(generated_all, title + 'English','generated_prob')

lang = 'AE'    
child_trans = pd.read_csv('Output/' + lang + '/production/ppl/generated/calculated/Calculated.csv')
plot_all(child_trans,lang)
   
def plot_distr(prompt_all):
    input_list = prompt_all['month'].tolist()
    unique_elements, counts = np.unique(input_list, return_counts=True)
    plt.bar(unique_elements, counts)
    plt.xlabel('Element')
    plt.ylabel('Count')
    
    if lang == 'AE':
        title = 'American '
    elif lang == 'BE':
        title = 'British '
        
    plt.title('Month Distribution of ' + title + 'English')
    plt.show()
'''
lang = 'BE'
text_path = 'Output/' + lang + '/production/ppl/generated'
prompt_all = pd.read_csv(text_path + '/Prompt_clean.csv')
plot_distr(prompt_all)
'''
def main(argv):
    
    # Args parser
    args = parseArgs(argv)
    
    lang = args.lang
    text_path = 'Output/' + lang + '/production/ppl/generated'
    
   
    # step 1: load data
    if not os.path.exists(text_path + '/Prompt_clean.csv'):     
        prompt_all = load_data(text_path)
        
    else:
        prompt_all = pd.read_csv(text_path + '/Prompt_clean.csv')
    
    # step 2: calculate the results
    # remove duplicated rows in all conditions
    columns_to_check = ['month', 'text_cleaned', 'prompt_cleaned']
    # Remove rows with duplicated values in the specified columns
    prompt_all = prompt_all[~prompt_all.duplicated(subset=columns_to_check, keep=False)].iloc[:30000]
    frame_all = get_prob(prompt_all)
    frame_all.to_csv(text_path + '/calculated/Prompt_calculated1.csv')
    
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
