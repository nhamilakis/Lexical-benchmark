#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
get recep scores; annotate different groups: filter each band's words

output: concatenated scores with annotated orth form and freq bands
@author: jliu

/scratch2/jliu/STELAWord/eval/recep/wuggy/score/200h/00/Machine/speech/score_lexical_test_all_trials.csv

intersections with the tested words 
'''
import os
import sys
import pandas as pd
import argparse


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Get recep vocab scores')
    
    parser.add_argument('--gold_path', type=str, default = '/scratch2/mlavechin/BabySLM/evaluation_sets/lexical/test/gold.csv',
                        help='Path to the wuggy material')
    
    parser.add_argument('--score_wuggy', type=str, default = '/scratch2/jliu/STELAWord/eval/recep/wuggy/',
                        help='root Path to final scores')
    
    parser.add_argument('--modality', type=str, default = 'phoneme',
                        help='modalities: phoneme or speech')
    
    parser.add_argument('--voice_thre', type=float, default = 0.75,
                        help='modalities: phoneme or speech')
    
    parser.add_argument('--word_thre', type=float, default = 0.8,
                        help='modalities: phoneme or speech')
    
    parser.add_argument('--pair_num', type=int, default = 5,
                        help='the number of phoneme pairs to compare')
    
    parser.add_argument('--band_num', type=int, default = 6,
                        help='the number of frequency bands')
    
    return parser.parse_args(argv)



def match_orth(score_dir, gold, modality,testset):
    
    
    def match_single_orth(score_path,gold):
        
        '''
        input:
            1.the score dataframe with each pair of word
            2.gold file with orthographic info
            
        return 
            the scoreframe with additional orth column
        '''
        
        score = pd.read_csv(score_path)
        # match the orthographic form with id
        score_grouped = score.groupby('id')
        
        score_frame = pd.DataFrame()
        for word_id,id_group in score_grouped: 
            id_group['orth'] = list(set(gold[gold['id'] == word_id]['word']))[0] 
            score_frame = pd.concat([score_frame,id_group])
        
        
        return score_frame
    
    score_frame_all = pd.DataFrame()
    # loop over the directory
    for dir_name in os.listdir(score_dir):
        if (dir_name != 'speech') and (dir_name != 'phoneme'):
            score_path = score_dir + '/' + dir_name + '/' + modality +'/score_lexical_test_all_trials.csv'
            
            if os.path.exists(score_path):
                score_frame = match_single_orth(score_path, gold)
                score_frame_all = pd.concat([score_frame_all,score_frame])
                
    # filter words in the testset
    score_frame_all = score_frame_all[score_frame_all['orth'].isin(testset['word'])]
    return score_frame_all


def match_band(score_frame,testset):
    
    '''
    input:
        1.the scoreframe with additional orth column
        2.the reference test set with the freq bands
        
    return 
        the scoreframe with additional freq band
   
    '''
    
    score_frame_grouped = score_frame.groupby('orth')
    
    # get the number of freq bands from test set
    band_num = len(set(testset['group']))
    freq_frame = pd.DataFrame()
    
    for orth,orth_group in score_frame_grouped:
        try:
            group = list(set(testset[testset['word'] == orth]['group']))[0] 
            if group != 'nan':
                
                orth_group['freq_' + str(band_num)] = group
            
        except: 
            pass      # partial result of the testset words
            
        freq_frame = pd.concat([freq_frame,orth_group])

    return freq_frame



def get_score(freq_frame,modality,voice_thre, word_thre,pair_num,band_num):
    
    '''
    Test the acoustic-phoneme representation mapping.

    Parameters:
    - freq_frame: The score frame annotated with freq band.
    - modality: The modality, e.g., 'speech'.
    - voice_thre: Voice threshold to decide whether a word is 'known'.
    - word_thre: Word threshold.
    - pair_num: The number of target pairs.

    Returns:
    - DataFrame with averaged scores by freq bands.
    '''
    
    def assign_binary(group,threshold):
        mean_value = group.mean()
        return 1 if mean_value > threshold else 0
    
    # sort by orthographic form and voice for replicability
    freq_frame = freq_frame.sort_values(by=['orth','voice'])
    
    # truncate the scoreframe into target pair_num; 
    freq_frame_grouped = freq_frame.groupby('orth')    
    selected_score_frame = pd.DataFrame()
    
    for orth,orth_group in freq_frame_grouped:
        # only select the target pair_num
        selected_id = list(set(orth_group['id']))[:pair_num]
        selected_orth_group = orth_group[orth_group['id'].isin(selected_id)]
        selected_score_frame = pd.concat([selected_score_frame,selected_orth_group])
    
    
    # get the average score by freq band 
    mean_score_frame = pd.DataFrame()
    
    # group by word's orthographical format and voice 
    selected_freq_group = selected_score_frame.groupby(['orth','voice'])
        
    for orth,orth_group in selected_freq_group:
            
        single_score = assign_binary(orth_group['score'],word_thre)
        # append score to first line of the orth group
        single_score_frame = orth_group.head(1)
        single_score_frame['phon_score'] = single_score
        mean_score_frame = pd.concat([mean_score_frame,single_score_frame])
    
    
    
    # apply voice threshold
    if modality == 'speech':
        score_frame = mean_score_frame
        mean_score_frame = pd.DataFrame()
        # update the raw score based on different voice group
        selected_freq_group = score_frame.groupby(['id'])
        for orth,orth_group in selected_freq_group:
                
            single_score = assign_binary(orth_group['phon_score'],voice_thre)
            # append score to first line of the orth group
            single_score_frame = orth_group.head(1)
            single_score_frame['phon_score'] = single_score
            mean_score_frame = pd.concat([mean_score_frame,single_score_frame])
    
    
    # average by freq bands
    grouped_frame = mean_score_frame.groupby(['freq_' + str(band_num)])['phon_score']
    final_frame = grouped_frame.mean().reset_index()
    final_frame['word_num'] = grouped_frame.size()
    final_frame.rename(columns={'freq_'+ str(band_num):'group','phon_score':'mean_score'}, inplace=True)
    
    return final_frame




def main(argv):
    
    # Args parser
    args = parseArgs(argv)
    
    
    score_root = args.score_wuggy + 'score/'
    score_test = args.score_wuggy + 'test_words/freq_' + str(args.band_num) + '/'
    gold = pd.read_csv(args.gold_path,encoding='unicode_escape')
    
    result_path = args.score_wuggy + 'results/freq_' + str(args.band_num) + '/'+ args.modality + '/' + str(args.voice_thre) + '_' + str(args.word_thre) + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    
    # concatenate the frames from the evluation sets
    # remove nan
    gold = gold.dropna()
    
    # loop over the testset folder
    for file in os.listdir(score_test):
        
        testset_path = score_test + file
        testset = pd.read_csv(testset_path)
    
        final_score_frame = pd.DataFrame()
        # loop over hour and chunk 
        for hour in os.listdir(score_root):
            
            for chunk in os.listdir(score_root + hour + '/'):
                
                score_dir = score_root + hour + '/' + chunk
                
                print('Calculating scores from ' + score_dir)
                
                try:

                    score_frame_all = match_orth(score_dir, gold, args.modality,testset)
                    
                    freq_frame = match_band(score_frame_all,testset)
                    freq_path = score_dir +  '/' + args.modality + '/' 
                    if not os.path.exists(freq_path):
                        os.makedirs(freq_path)
                       
                    freq_frame.to_csv(freq_path + file)
                     
                    
                    #freq_frame = pd.read_csv(freq_path + file)
                    mean_score_frame = get_score(freq_frame,args.modality,args.voice_thre, args.word_thre,args.pair_num,args.band_num)
                    
                    # save the result to the target dir
                    mean_score_frame['hour'] = hour
                    mean_score_frame['month'] = int(hour[:-1])/89
                    mean_score_frame['chunk'] = chunk
                    
                    final_score_frame = pd.concat([final_score_frame,mean_score_frame])
                    
                except:
                   print(score_dir)
        final_score_frame.to_csv(result_path + file)  
            
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)


