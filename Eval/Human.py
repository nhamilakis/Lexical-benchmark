#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read CHILDES scripts adnd return the frequency-based curve

@author: jliu
"""
import os
from util import load_transcript, get_freq, count_by_month, calculate_fitness
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns   
from scipy.interpolate import interp1d



def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Investigate CHILDES corpus')
    
    parser.add_argument('--lang', type=str, default = 'AE',
                        help='langauges to test: AE, BE or FR')
    
    parser.add_argument('--TextPath', type=str, default = 'CHILDES',
                        help='root Path to the CHILDES transcripts; one of the variables to invetigate')
    
    parser.add_argument('--OutputPath', type=str, default = 'Output',
                        help='Path to the freq output.')
    
    parser.add_argument('--condition', type=str, default = 'recep',
                        help='types of vocab: recep or exp')
    
    parser.add_argument('--hour', type=int, default = 3,
                        help='the estimated number of hours per day')
    
    parser.add_argument('--word_per_hour', type=int, default = 10000,
                        help='the estimated number of words per hour')
    
    parser.add_argument('--threshold_range', type=list, default = [10],
                        help='threshold to decide knowing a productive word or not, one of the variable to invetigate')
    
    parser.add_argument('--eval_path', type=str, default = 'Human_CDI/',
                        help='path to the evaluation material; one of the variables to invetigate')
    
    return parser.parse_args(argv)






def load_data(TextPath,OutputPath,lang,condition):
    
    '''
    get word counts from the cleaned transcripts
    input: text rootpath with CHILDES transcripts
    output: freq_frame, score_frame
    '''
    
    # load and clean transcripts: get word info in each seperate transcript
    if os.path.exists(OutputPath + '/stat_per_file_' + lang +'.csv'):     
        print('The transcripts have already been cleaned! Skip')
        file_stat_sorted = pd.read_csv(OutputPath + '/stat_per_file_' + lang +'.csv')
        
    # check whether the cleaned transcripts exist
    else:
        print('Start cleaning files')
        file_stat = pd.DataFrame()
        for lang in os.listdir(TextPath):  
            output_dir =  OutputPath + '/' + lang
            for file in os.listdir(TextPath + '/' + lang): 
                try: 
                    file_frame = load_transcript(TextPath,output_dir,file,lang,condition)
                    file_stat = pd.concat([file_stat,file_frame])
                    
                except:
                    print(file)
                    
        file_stat_sorted = file_stat.sort_values('month')    
        file_stat_sorted.to_csv(OutputPath + '/stat_per_file_' + lang +'.csv')                
        print('Finished cleaning files')
    
    # concatenate word info in each month
    month_stat = count_by_month(OutputPath,file_stat_sorted)
    
    return month_stat




def count_words(OutputPath,group_stat,eval_path,hour,word_per_hour):
    
    for file in os.listdir(eval_path):
        eval_lst = pd.read_csv(eval_path + '/' + file)['words'].tolist()
        
        
    freq_frame = pd.DataFrame()
    freq_frame['word'] = eval_lst

    # loop each month
    for file in group_stat['group'].tolist():
        
        # get word freq list for each file
        text_file = 'transcript_' + file.split('.')[0].split('-')[1]
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

        fre_table = get_freq(word_lst)
        
        freq_lst = []
        for word in eval_lst:
            try: 
                # recover to the actual count
                
                norm_count = fre_table[fre_table['Word']==word]['Norm_freq'].item() * 30 * word_per_hour * hour
                
            except:
                norm_count = 0
            freq_lst.append(norm_count)
        freq_frame[file] = freq_lst
        
    
    # we use cum freq as the threshold for the word
    sel_frame = freq_frame.iloc[:,1:]
    columns = freq_frame.columns[1:]
    sel_frame = sel_frame.cumsum(axis=1)
            
    for col in columns.tolist():
        freq_frame[col] = sel_frame[col] 
    
    
    freq_frame.to_csv(OutputPath + '/selected_freq.csv')
    
    return freq_frame



def get_score(freq_frame,OutputPath,threshold,hour):
    
    '''
    get scores of the target words in Wordbank 
    input: word counts in each chunk/month and the stat of the number of true words as well as the true data proportion
    output: a dataframe with each word count        
    '''
     
    score_frame = pd.DataFrame()
    
    
    # get each chunk's scores based on the threshold
    columns = freq_frame.columns[1:]
      
    for col in columns.tolist():
        score_lst = []
        
        for count in freq_frame[col].tolist():
            
            # varying thresholds based on the correction factors
            if count >= threshold:
                score = 1
            else:
                score = 0
                        
            score_lst.append(score)
                
        score_frame[col] = score_lst
    
    score_path = OutputPath + '/Scores/'
    if not os.path.exists(score_path):
        os.makedirs(score_path) 
    score_frame.to_csv(score_path + '/score_' + str(threshold) +'.csv')
    return score_frame
    


def plot_curve(OutputPath,eval_path,score_frame,threshold,group_stat,condition):
    
    
    sns.set_style('whitegrid') 
    
    column_list = score_frame.columns.tolist()[2:]
    
    for chunk in column_list: 
        min_month = group_stat[group_stat['group'] == chunk]['start_month'].item()
        max_month = group_stat[group_stat['group'] == chunk]['end_month'].item()
        column_name = str(min_month) + '-' + str(max_month)
        score_frame.rename(columns={chunk: column_name}, inplace=True)     
        
    # get the average-frame 
    column_averages = score_frame.mean()[1:]
    
    # Columns to plot
    columns_to_plot = column_averages.index
    
    # Customized bar widths
    
    labels = []
    
    for column in columns_to_plot:
        column_left = column.split('-')[0]
        
        labels.append(int(column_left))
        
    # get the average value of the same bar
    df = pd.DataFrame([labels, column_averages.tolist()]).T
    df = df.rename(columns={0: "month", 1: "Value"})
    
    
    # Group by 'Group' column and calculate the mean of 'Value' for each group
    '''
    get the average value of the same bar/month
    '''
    mean_values = df.groupby('month')['Value'].mean()
    plt.plot(mean_values, label='CHILDES')
    
    
    # plot the CDI results
    for file in os.listdir(eval_path):
        selected_words = pd.read_csv(eval_path + '/' + file).iloc[:, 5:-4]
        
    
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
    
    ax = sns.lineplot(x="month", y="Proportion of acquired words", data=data_frame_final, label='CDI')
    
    # set the limits of the x-axis for each line
    for line in ax.lines:
        plt.xlim(0,36)
        plt.ylim(0,1)
    
    
    plt.title('CHILDES {} vocab (threshold = {})'.format(condition,threshold))
    plt.xlabel('age in month', fontsize=15)
    plt.ylabel('Proportion of children', fontsize=15)

    plt.tick_params(axis='both', labelsize=10)
  
    plt.legend()
    figure_path = OutputPath + '/Figures/'
    if not os.path.exists(figure_path):
        os.makedirs(figure_path) 
    plt.savefig(figure_path + '/Curve_'+ str(threshold) + '.png')
    plt.show()
    
    
    mean_value_CDI = data_frame_final.groupby('month')['Proportion of acquired words'].mean()
    fitness = calculate_fitness(mean_values, mean_value_CDI)
    return fitness
      

def main(argv):
    
    # Args parser
    args = parseArgs(argv)
    
    TextPath = args.TextPath
    condition = args.condition
    lang = args.lang
    OutputPath = args.OutputPath + '/' + lang + '/' + condition
    eval_path = args.eval_path + lang + '/' + condition
    hour = args.hour
    threshold_range = args.threshold_range
    word_per_hour = args.word_per_hour
    rmse_frame_all = pd.DataFrame()
    
    # step 1: load data and count words
    month_stat = load_data(TextPath,OutputPath,lang,condition)
    freq_frame = count_words(OutputPath,month_stat,eval_path,hour,word_per_hour)
    
    # step 2: get the score based on different thresholds
    for threshold in threshold_range:
    
        score_frame = get_score(freq_frame,OutputPath,threshold,hour)
             
        # plot the developmental bars and calculate the fitness
        print('Plotting the developmental bars')
             
        # compare and get the best combination of threshold two variables
        rmse = plot_curve(OutputPath,eval_path,score_frame,threshold,month_stat,condition)
        rmse_frame_temp = pd.DataFrame([threshold, rmse]).T
        rmse_frame = rmse_frame_temp.rename(columns={0: "Chunksize", 1: "threshold", 2: "rmse" })    
        rmse_frame_all = pd.concat([rmse_frame_all,rmse_frame])
        
       
    rmse_frame_all.to_csv(OutputPath + '/Scores/Fitness_All.csv')  
    
    

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    
    