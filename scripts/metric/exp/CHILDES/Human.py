#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read CHILDES scripts adnd return the frequency-based curve

between 6 and 36 the following nb of seconds per hours


@author: jliu
"""
import os
from util import load_transcript, get_freq, count_by_month, calculate_fitness,get_score
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns   



def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Investigate CHILDES corpus')
    
    parser.add_argument('--lang', type=str, default = 'BE',
                        help='langauges to test: AE, BE or FR')
    
    parser.add_argument('--eval_type', type=str, default = 'CDI',
                        help='langauges to test: CDI or Wuggy_filtered')
    
    parser.add_argument('--eval_condition', type=str, default = 'exp',
                        help='which type of words to evaluate; recep or exp')
    
    parser.add_argument('--TextPath', type=str, default = 'CHILDES',
                        help='root Path to the CHILDES transcripts; one of the variables to invetigate')
    
    parser.add_argument('--OutputPath', type=str, default = 'Output',
                        help='Path to the freq output.')
    
    parser.add_argument('--input_condition', type=str, default = 'exp',
                        help='types of vocab: recep or exp')
    
    parser.add_argument('--hour', type=dict, default = 10,
                        help='the estimated number of waking hours per day; data from Alex(2023)')
    
    parser.add_argument('--word_per_sec', type=int, default = 3,
                        help='the estimated number of words per second')
    
    parser.add_argument('--sec_frame_path', type=str, default = 'vocal_month.csv',
                        help='the estmated vocalization seconds per hour by month')
    
    parser.add_argument('--threshold_range', type=list, default = [50,100,200,300],
                        help='threshold to decide knowing a productive word or not, one of the variable to invetigate')
    
    parser.add_argument('--eval_path', type=str, default = 'Human_eval/',
                        help='path to the evaluation material; one of the variables to invetigate')
    
    return parser.parse_args(argv)



def load_data(TextPath,OutputPath,lang,input_condition):
    
    '''
    get word counts from the cleaned transcripts
    input: text rootpath with CHILDES transcripts
    output: freq_frame, score_frame
    '''
    
    # load and clean transcripts: get word info in each seperate transcript
    if os.path.exists(OutputPath + '/stat_per_file' +'.csv'):     
        print('The transcripts have already been cleaned! Skip')
        file_stat_sorted = pd.read_csv(OutputPath + '/stat_per_file' +'.csv')
        
    # check whether the cleaned transcripts exist
    else:
        print('Start cleaning files')
        file_stat = pd.DataFrame()
        for lang in os.listdir(TextPath):  
            output_dir =  OutputPath + '/' + lang
            for file in os.listdir(TextPath + '/' + lang): 
                try: 
                    file_frame = load_transcript(TextPath,output_dir,file,lang,input_condition)
                    file_stat = pd.concat([file_stat,file_frame])
                    
                except:
                    print(file)
                    
        file_stat_sorted = file_stat.sort_values('month')    
        file_stat_sorted.to_csv(OutputPath + '/stat_per_file' +'.csv')                
        print('Finished cleaning files')
    
    # concatenate word info in each month
    month_stat = count_by_month(OutputPath,file_stat_sorted)
    
    return month_stat



def count_words(OutputPath,group_stat,eval_path,hour,word_per_sec,eval_type,lang,eval_condition,sec_frame):
    
    '''
    count words of the given list
    
    '''
    
    eval_dir = eval_path + eval_type + '/' + lang + '/' + eval_condition
    
    
           
    for file in os.listdir(eval_dir):
        eval_frame = pd.read_csv(eval_dir + '/' + file)
        eval_lst = eval_frame['word'].tolist()
        
    freq_frame = pd.DataFrame()
    freq_frame['word'] = eval_lst
    freq_frame['group_original'] = eval_frame['group_original'].tolist()
    
    # loop each month
    for file in set(group_stat['end_month'].tolist()):
        
        # get word freq list for each file
        text_file = 'transcript_' + str(file)
        file_path = OutputPath + '/Transcript_by_month/' + text_file + '.txt'  
        
        with open(file_path, encoding="utf8") as f:
            sent_lst = f.readlines()
        word_lst = []    
        for sent in sent_lst:
            # remove the beginning and ending space
            words = sent.split(' ')
            
            for word in words:
                cleaned_word = word.strip()
                if len(cleaned_word) > 0:  
                    word_lst.append(cleaned_word)
        # save the overall freq dataframe for further use
        fre_table = get_freq(word_lst)
        
        
        
        freq_lst = []
        for word in eval_lst:
            try: 
                # recover to the actual count based on Alex's paper
                
                sec_per_hour = sec_frame[sec_frame['month']==file]['sec_per_hour'].item()
                norm_count = fre_table[fre_table['Word']==word]['Norm_freq'].item() * 30 * word_per_sec * hour * sec_per_hour
                
            except:
                norm_count = 0
            freq_lst.append(norm_count)
        freq_frame[file] = freq_lst
        
    # sort the target frameRecep vocab
    
    # we use cum freq as the threshold for the word
    sel_frame = freq_frame.iloc[:,2:]
    columns = freq_frame.columns[2:]
    sel_frame = sel_frame.cumsum(axis=1)
            
    for col in columns.tolist():
        freq_frame[col] = sel_frame[col] 
    
    
    freq_frame.to_csv(OutputPath + '/selected_freq.csv')
    
    return freq_frame




def plot_multiple(OutputPath,eval_path,threshold_range,group_stat,eval_condition,freq_frame,hour,lang,eval_type):
    
    
    sns.set_style('whitegrid') 
    
    
    eval_dir = eval_path + 'CDI' + '/' + lang + '/' + eval_condition   
   
    # plot the CDI results
    # load multiple files
    for file in os.listdir(eval_dir):
        selected_words = pd.read_csv(eval_dir + '/' + file).iloc[:, 5:-6]
        
    
    size_lst = []
    month_lst = []
    
    n = 0
    while n < selected_words.shape[0]:
        size_lst.append(selected_words.iloc[n])
        headers_list = selected_words.columns.tolist()
        month_lst.append(headers_list)
        n += 1

    size_lst_final = [item for sublist in size_lst for item in sublist]
    month_lst_final = [item for sublist in month_lst for item in sublist]
    month_lst_transformed = []
    for month in month_lst_final:
        month_lst_transformed.append(int(month))
    # convert into dataframe
    data_frame = pd.DataFrame([month_lst_transformed,size_lst_final]).T
    data_frame.rename(columns={0:'month',1:'Proportion of acquired words'}, inplace=True)
    data_frame_final = data_frame.dropna(axis=0)
    
    
    ax = sns.lineplot(x="month", y="Proportion of acquired words", data=data_frame_final, color='black', linewidth=2.5, label=lang + '_CDI')
    
    # set the limits of the x-axis for each line
    for line in ax.lines:
        plt.xlim(0,36)
        plt.ylim(0,1)
    mean_value_CDI = data_frame_final.groupby('month')['Proportion of acquired words'].mean()
    
    
    
    # plot the model results
    rmse_frame_all = pd.DataFrame()
    # loop over thresholds
    for threshold in threshold_range:
        
        avg_values_lst = []
        # averaged by different groups
        for freq in set(list(freq_frame['group_original'].tolist())):
            
            word_group = freq_frame[freq_frame['group_original']==freq]
            score_frame,avg_value = get_score(word_group,OutputPath,threshold,hour)
            avg_values_lst.append(avg_value.values)
        
        # problem here: only for the high freq words in the end
        #avg_values = (avg_values_lst[0] + avg_values_lst[1]+avg_values_lst[2] + avg_values_lst[3]+avg_values_lst[4] + avg_values_lst[5]) / 6    
        
        avg_values = (avg_values_lst[0] + avg_values_lst[1]) / 2
        # Plotting the line curve
        ax = sns.lineplot(score_frame.columns, avg_values, label= 'threshold: ' + str(threshold))
    '''    
        # convert back to series to calculate fitness fo the two curves
        mean_value_CHILDES = pd.Series(avg_values, index=score_frame.columns)
        rmse = calculate_fitness(mean_value_CHILDES, mean_value_CDI)
        rmse_frame_temp = pd.DataFrame([threshold, rmse]).T
        rmse_frame = rmse_frame_temp.rename(columns={0: "Chunksize", 1: "threshold", 2: "rmse" })    
        rmse_frame_all = pd.concat([rmse_frame_all,rmse_frame])
    '''    
    plt.title('{} CHILDES {} vocab(tested on {})'.format(lang,eval_condition,eval_type), fontsize=15)
    #plt.title('Accumulator on {} CHILDES ({} vocab)'.format(lang,eval_condition), fontsize=15)
    plt.xlabel('age in month', fontsize=15)
    plt.ylabel('Proportion of children', fontsize=15)
    
    
    plt.tick_params(axis='both', labelsize=10)
  
    plt.legend()
    
    figure_path = OutputPath + '/Figures/'
    if not os.path.exists(figure_path):
        os.makedirs(figure_path) 
    plt.savefig(figure_path + '/Curve.png',dpi=800)
    plt.show()
    
    #rmse_frame_all.to_csv(OutputPath + '/Scores/Fitness_All.csv')  
    
    
    

def main(argv):
    
    # Args parser
    args = parseArgs(argv)
    
    TextPath = args.TextPath
    eval_condition = args.eval_condition
    input_condition = args.input_condition
    lang = args.lang
    eval_type = args.eval_type
    OutputPath = args.OutputPath + '/' + eval_type + '/' + lang + '/' + eval_condition
    eval_path = args.eval_path
    hour = args.hour
    threshold_range = args.threshold_range
    word_per_sec = args.word_per_sec
    sec_frame = pd.read_csv(args.sec_frame_path)
        
    # step 1: load data and count words
    month_stat = load_data(TextPath,OutputPath,lang,input_condition)
    freq_frame = count_words(OutputPath,month_stat,eval_path,hour,word_per_sec,eval_type,lang,eval_condition,sec_frame)
                            
    # step 2: get the score based on different thresholds
    plot_multiple(OutputPath,eval_path,threshold_range,month_stat,eval_condition,freq_frame,hour,lang,eval_type)
    

   

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    


