#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refined pipeline for wuggy test
Recursively run the results

@author: jliu

"""
import os
import sys
import pandas as pd
import argparse
import subprocess
import logging



def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Investigate CHILDES corpus')
    
    parser.add_argument('--Test_path', type=str, default = 'test_words',
                        help='Path to the test vocab and the wuggy material')
    
    parser.add_argument('--score_root', type=str, default = 'score',
                        help='root Path to final scores')
    
    parser.add_argument('--model_root', type=str, default = 'model',
                        help='root Path to model checkpoints ')
    
    parser.add_argument('--modality', type=str, default = 'speech',
                        help='modalities of teasted files; phoneme or speech')
    
    parser.add_argument('--data_root', type=str, default = 'model',
                        help='root Path to data checkpoints ')
    
    parser.add_argument('--audio_test', type=str, default = 'test_audio',
                        help='Path to the audio set')
    
    parser.add_argument('--word_set', default = 'CDI',
                        help='what word to test')
    
    parser.add_argument('--gpu', default = True,
                        help='whether to use gpu')
    
    parser.add_argument('--resume', default = True,
                        help="Continue to compute score if the output file already exists.")
    
    parser.add_argument('--debug', default = True,
                        help="Load only a very small amount of files for "
                        "debugging purposes.")
    
    return parser.parse_args(argv)



def run_command(command):
    subprocess.call(command, shell=True)


'''
step 1: select the target word set

test the gold.csv respectively and decide whether to merge them
'''


def load_data(TestPath,MaterialPath,modality):
    
    '''
    load the .csv containing the correspondent word list and output the .txt test file
    the phoneme modality contain the merged file; 
    speech modelaity: file nam
    '''
    
    test = pd.read_csv(TestPath)
    
    material = pd.read_csv(MaterialPath)
    # match the words
    test_ids = material[material['word'].isin(
            test['word'])]['id'].unique()
    
    selected_test = material[material['id'].isin(test_ids)]
    
    # merge the results for phoneme modality
    if modality == 'phoneme':
        voices = selected_test['voice'].unique()
        selected_test = selected_test[selected_test['voice'] == voices[0]]
    
    # convert into the required format
    def insert_commas(text):
        return text.replace(' ', ',')
    
    
    selected_test['phones'] = selected_test['phones'].apply(insert_commas)
    # output the .txt file
    if modality == 'speech':  # only preserve filenames for speech quantization
        selected_material = selected_test[['filename']]
    else:
        selected_material = selected_test[['filename', 'phones']]
    
    # Export selected columns to a .txt file separated by tabs
    selected_material.to_csv('test.txt', sep='\t', index=False, header=False)
    
    selected_test.to_csv('gold.csv', index=False)
    
    return selected_material



'''
step 1.5: quantize the tested audios with trained models
TO DO: add gpu setting
'''

quantize_audio_temp = 'python quantize_audio.py --pathClusteringCheckpoint {model_root}/{hour}/{chunk}/kmeans50/checkpoint_last.pt \
    --pathDB {audio_test} \
    --pathOutputDir {score_root}/{hour}/{chunk}/{word_set}/{modality} \
    --pathSeq {test_path}/{test}.txt \
    --output_filename {test} \
    --gpu {gpu_status} \
    --resume {resume}'



'''
step 2: compute the prob

phoneme modality: one test file outside the directory
speech modality: in the corresponding subfolders

name of the dictionary files are different! 
'''

compute_prob_temp = 'python compute_prob.py --pathQuantizedUnits {test_path}/{test}.txt \
    --pathOutputFile {score_root}/{hour}/{chunk}/{word_set}/{modality}/{log_prob}.txt \
    --pathLSTMCheckpoint {model_root}/{hour}/{chunk}/{checkpoint_parent}/checkpoint_best.pt \
    --dict {data_root}/{hour}/{chunk}/{dict_parent}/dict.txt \
    --gpu {gpu_status} \
    --resume {resume}'



'''
step 3: get the scores recursively

!!! change this 
'''

compute_lexical_temp = 'python compute_lexical.py \
    -o {score_root}/{hour}/{chunk}/{word_set} \
    -g {test_path}/{gold}.csv \
    -p {score_root}/{hour}/{chunk}/{word_set}/{modality}/{log_prob}.txt \
    -k test \
    --modality {modality}'




def main(argv):
    
    # Configure the logging
    logging.basicConfig(filename='error.log', level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Args parser
    args = parseArgs(argv)
    
    if not os.path.exists(args.score_root):
        os.makedirs(args.score_root)
    
    checkpoint_parent = 'checkpoints'
    dict_parent = 'bin_data'
    test_arg = 'test'
    log_prob_arg = 'log_prob'
    gold_arg = 'gold'
    
    if args.debug:
        test_arg += '_debug'
        log_prob_arg += '_debug'
        gold_arg += '_debug'
    
    # check whether exists the material for testing; if not, generate ones      
    
    if not os.path.exists(args.Test_path + '/' + args.word_set + '/' + args.modality + '/gold.csv') or not os.path.exists(args.Test_path + '/' + args.word_set + '/' + args.modality + '/test.txt'):     
        print('Selecting {} words for evluation'.format(args.word_set))
        load_data(args.Test_path + '/' + args.word_set + '/CDI.csv',args.Test_path+ '/' + args.word_set + '/gold.csv',args.modality)
    
   
    # loop over model root to ensure that the trained model exists
    hour_lst = ['3200h']
    chunk_lst = ['00']
    
    '''
    for hour in os.listdir(args.model_root): 
        
        for chunk in os.listdir(args.model_root + '/' + hour): 
    '''
    for hour in hour_lst: 
        
        for chunk in chunk_lst: 
        
            # create the output directory 
            output_path = '{score_root}/{hour}/{chunk}/{word_set}'.format(score_root = args.score_root,hour = hour, chunk = chunk
                                                                          ,word_set=args.word_set)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
                
            test_path = args.Test_path + '/' + args.word_set + '/' + args.modality
            
            if args.modality == 'speech':
                dict_parent = 'lstm'
                checkpoint_parent = 'lstm'
                
                quantize_audio = quantize_audio_temp.format(score_root = args.score_root,hour = hour, chunk = chunk, model_root = args.model_root
                                                            ,audio_test = args.audio_test, word_set=args.word_set,resume = args.resume
                                                            ,modality = args.modality,test_path = test_path,test = test_arg, gpu_status = args.gpu)
                                                            
                run_command(quantize_audio)
                # the quantized file should be in each folder
                test_path = '{score_root}/{hour}/{chunk}/{word_set}/{modality}'.format(score_root = args.score_root,hour = hour, chunk = chunk
                                                           ,word_set=args.word_set,modality = args.modality)
                
            compute_prob = compute_prob_temp.format(score_root = args.score_root,hour = hour, chunk = chunk, model_root = args.model_root
                                                        ,data_root = args.data_root, gpu_status = args.gpu,word_set=args.word_set
                                                        ,modality = args.modality,resume = args.resume,test_path = test_path
                                                        ,test = test_arg, log_prob = log_prob_arg,dict_parent = dict_parent
                                                        ,checkpoint_parent = checkpoint_parent)
            
            
            compute_lexical = compute_lexical_temp.format(score_root = args.score_root,hour = hour, chunk = chunk,word_set=args.word_set
                                                        ,modality = args.modality, log_prob = log_prob_arg, gold = gold_arg,test_path = args.Test_path + '/' + args.word_set + '/' + args.modality)
            
            
            
            print('Extracting log prob from: {hour}/{chunk}'.format(hour = hour, chunk = chunk))
            try:  
                   
                run_command(compute_prob)
                
                
                print('Computing lexical scores from: {hour}/{chunk}'.format(hour = hour, chunk = chunk))     
                try:
                    run_command(compute_lexical)
            
                except:
                    # Log the error
                    logging.error(f"Fail to compute prob from: {hour}/{chunk}".format(hour = hour, chunk = chunk))  
           
            except:
                # Log the error
                
                logging.error(f"Fail to extract prob from: {hour}/{chunk}".format(hour = hour, chunk = chunk))     
                
            
            
              


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    
    
