
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script to evaluate hyperparameters
1.compare entropy and log prob distr of the train data and genrated prompts
2. count the numbwer of generated words 
@author: jliu
"""


import os
import pandas as pd
from KL_util import mean_KL,match_seq,lemmatize
import collections
import sys
import argparse


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Generate tokens from decoder models')
    
    
    parser.add_argument('--number', default = '1',
                        help='a list of top-k or top_p candidates')
    
    parser.add_argument('--root_path', default = 'KL_material/',
                        help='a list of temperature parameters for optimization')
    
    parser.add_argument('--output_path', default = 'KL_eval/',
                        help='the output path of the calculated scores')
    
    parser.add_argument('--KL', type=bool, default = True,    # to be modified later!!!
                        help='a list of temperature parameters for optimization')
    
    parser.add_argument('--gpu', type=bool, default = True,
                        help= 'whether to use gpu')
   
    return parser.parse_args(argv)




def get_distr(root_path,gpu,KL):
    
    '''
    get the entropy distr and KL divergence with the reference data for each type 
    
    input: the root directory containing all the generated files adn train reference
    output: 1.the info frame with all the generarted tokens
            2.the reference frame with an additional column of the month info
            3.vocab size frame with the seq word and lemma frequencies
    '''
    
    # load the rerference data
    frame_all = []
    seq_all = []
    month_lst = []
    prompt_lst = []
    h_all = []
    prob_all =[]
    strategy_lst = []
    beam_lst = []
    topk_lst = []
    topp_lst = []
    random_lst = []
    temp_lst = []
    h_dist_lst = []
    prob_dist_lst = []
    directory_lst = []
    reference_frame = pd.DataFrame()
    
    # go over the generated files recursively
    
    for month in os.listdir(root_path): 
        if not month.endswith('.csv') and not month.endswith('.ods'): 
            train_distr = pd.read_csv(root_path + '/' + month + '/train_distr.csv')
            train_distr['month'] = month
            reference_frame = pd.concat([reference_frame,train_distr])
                    
            
            for prompt_type in os.listdir(root_path + '/' + month): 
                
                
                if not prompt_type.endswith('.csv') and not prompt_type.endswith('.ods'):                    
                    for strategy in os.listdir(root_path + '/' + month+ '/' + prompt_type): 
                        
                        for file in os.listdir(root_path + '/' + month+ '/' + prompt_type+ '/' + strategy):
                                  # in the case that entroyp and prob are not calculated yet
                                # load decoding strategy information
                                
                                data = pd.read_csv(root_path + '/' + month+ '/' + prompt_type + '/' + strategy + '/' + file)
                                
                                if not file.startswith('.'):
                                    
                                    try:
                                        
                                        # calculate the KL divergence between the reference and generated distr                                       
                                        if KL:
                                        
                                            h_dist = mean_KL(data['LSTM_generated_h'].tolist(),train_distr['entropy'].tolist(),gpu)
                                            prob_dist = mean_KL(data['LSTM_generated_prob'].tolist(),train_distr['prob'].tolist(),gpu)
                                            
                                            h_dist_lst.append(h_dist)
                                            prob_dist_lst.append(prob_dist)
                                            
                                            prob_all.append(data['LSTM_generated_prob'].tolist())
                                            h_all.append(data['LSTM_generated_h'].tolist())
                                            print('SUCCESSFUL KL')
                                        else:
                                    
                                            h_dist_lst.append(1)
                                            prob_dist_lst.append(1)
                                            
                                            prob_all.append(data['LSTM_generated_prob'].tolist())
                                            h_all.append(data['LSTM_generated_h'].tolist())
                                            print('FAILED KL')
                                            
                                        # count words
                                        seq = []
                                        n = 0
                                        while n < data.shape[0]:
                                            generated = data['LSTM_segmented'].tolist()[n].split(' ')
                                            seq.extend(generated)
                                            n += 1
                                        
                                        # get freq lists
                                        frequencyDict = collections.Counter(seq)  
                                        freq_lst = list(frequencyDict.values())
                                        word_lst = list(frequencyDict.keys())
                                        fre_table = pd.DataFrame([word_lst,freq_lst]).T
                                        
                                        col_Names=["Word", "Freq"]
                                        
                                        fre_table.columns = col_Names
                                        seq_all.extend(seq)
                                        
                                        
                                        if strategy == 'beam':
                                            beam_lst.append(file.split('_')[0])
                                            topk_lst.append('0')
                                            topp_lst.append('0')
                                            random_lst.append('0')
                                            strategy_lst.append(strategy)
                                            fre_table['BEAM'] = file.split('_')[0]
                                            fre_table['TOPK'] ='0'
                                            fre_table['TOPP'] ='0'
                                            fre_table['RANDOM'] ='0'
                                            fre_table['DECODING'] = strategy
                                            
                                        elif strategy == 'sample_topk':
                                            topk_lst.append(file.split('_')[0])
                                            beam_lst.append('0')
                                            topp_lst.append('0')
                                            random_lst.append('0')
                                            strategy_lst.append(strategy.split('_')[1])
                                            fre_table['TOPK'] = file.split('_')[0]
                                            fre_table['BEAM'] ='0'
                                            fre_table['TOPP'] ='0'
                                            fre_table['RANDOM'] ='0'
                                            fre_table['DECODING'] = strategy.split('_')[1]
                                            
                                            
                                        elif strategy == 'sample_topp':
                                            topp_lst.append(file.split('_')[0])
                                            beam_lst.append('0')
                                            topk_lst.append('0')
                                            random_lst.append('0')
                                            strategy_lst.append(strategy.split('_')[1])
                                            fre_table['TOPP'] = file.split('_')[0]
                                            fre_table['BEAM'] ='0'
                                            fre_table['TOPK'] ='0'
                                            fre_table['RANDOM'] ='0'
                                            fre_table['DECODING'] = strategy.split('_')[1]
                                            
                                        elif strategy == 'sample_random':
                                            random_lst.append('1')
                                            topk_lst.append('0')
                                            beam_lst.append('0')
                                            topp_lst.append('0')
                                            strategy_lst.append(strategy.split('_')[1])
                                            fre_table['RANDOM'] = '1'
                                            fre_table['BEAM'] ='0'
                                            fre_table['TOPP'] ='0'
                                            fre_table['TOPK'] ='0'
                                            fre_table['DECODING'] = strategy.split('_')[1]
                                            
                                        # concatnete all the basic info regarding the genrated seq
                                        fre_table['MONTH'] = month
                                        fre_table['PROMPT'] = prompt_type
                                        fre_table['TEMP'] = file.split('_')[1]
                                        
                                        prompt_lst.append(prompt_type)
                                        month_lst.append(month)
                                        temp_lst.append(float(file.split('_')[1]))
                                        directory_lst.append(month+ '/' + prompt_type + '/' + strategy + '/' + file)
                                        frame_all.append(fre_table)
                                        
                                        print('SUCCESS: ' + month+ '/' + prompt_type + '/' + strategy + '/' + file)    
                                    
                                    
                                    except:
                                        print('FAILURE: ' + month+ '/' + prompt_type + '/' + strategy + '/' + file)
                                
                                else:
                                    print(file)
    
    
    
    info_frame = pd.DataFrame([month_lst,prompt_lst,strategy_lst,beam_lst,topk_lst,topp_lst,random_lst,temp_lst,h_all,prob_all,prob_dist_lst,h_dist_lst,directory_lst]).T
    
    # rename the columns
    info_frame.rename(columns = {0:'month', 1:'prompt',2:'decoding', 3:'beam', 4:'topk', 5:'topp',6:'random', 7:'temp',8:'entropy',9:'prob',10:'prob_dist',11:'entropy_dist',12:'location'}, inplace = True)
    # remove the row with NA values
    
    info_frame = info_frame.dropna()
    info_frame = info_frame[(info_frame['random'] != '.') & (info_frame['topp'] != '.') & (info_frame['topk'] != '.') & (info_frame['prob'] != '[]')]
    
    
    # sort the result based on temperature to get more organized legend labels
    info_frame = info_frame.sort_values(by='temp', ascending=True)
    
    
    # # get word count and lemma count frames
    seq_lst = list(set(seq_all))
    
    seq_frame = match_seq(seq_lst,frame_all)
    
    word_lst, lemma_dict = lemmatize(seq_lst)
    
    word_lst.extend(['MONTH','PROMPT','BEAM','TOPK','TEMP','TOPP','RANDOM','DECODING'])
    word_frame = seq_frame[word_lst]
    
    # reshape the lemma frame based onthe word_frame: basic info, lemma, total counts
    lemma_frame = seq_frame[['MONTH','PROMPT','BEAM','TOPK','TEMP','TOPP','RANDOM','DECODING']]
    for lemma, words in lemma_dict.items():
        
        # Merge columns in the list by adding their values
        lemma_frame[lemma] = word_frame[words].sum(axis=1)
    
    return info_frame, reference_frame, seq_frame, word_frame, lemma_frame



def main(argv):
    
    
    # Args parser
    args = parseArgs(argv)
        
    gpu = args.gpu  
    KL = args.KL
    number = args.number
    output_path = args.output_path
    root_path = args.root_path + 'eval' + number
    
    
    info_frame, reference_frame, seq_frame, word_frame, lemma_frame = get_distr(root_path,gpu,KL)

    info_frame.to_csv(output_path + 'Info_frame' + number + '.csv')
    reference_frame.to_csv(output_path + 'Reference_frame' + number + '.csv')
    seq_frame.to_csv(output_path + 'seq_frame'+ number + '.csv')
    word_frame.to_csv(output_path + 'word_frame'+ number + '.csv')
    lemma_frame.to_csv(output_path + 'lemma_frame'+ number + '.csv')
    
    
    
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    

    