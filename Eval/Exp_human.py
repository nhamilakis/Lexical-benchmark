# -*- coding: utf-8 -*-
"""
get freq of CDI receptive vocab from CHILDES children's utterances and also fitness between accumulator and CDI data

input: 1.CHILDES transcript; 2.CDI data
output: 1.normalized words' frequency in CDI, based on children's utterances in CHILDES; 2. figures of accumulator and CDI

 
"""



import numpy as np
import os
from util import clean_text, chunk_list, get_freq
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns   
from scipy.interpolate import interp1d

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Investigate CHILDES corpus')
    
    parser.add_argument('--TextPath', type=str, default = 'CHILDES/BE',
                        help='root Path to the CHILDES transcripts; one of the variables to invetigate')
    
    parser.add_argument('--OutputPath', type=str, default = 'Output/BE',
                        help='Path to the freq output.')
    
    parser.add_argument('--Chunksize_range', type=list, default = [0.5],
                        help='a list of the smallest size for each chunk in hours')
    
    parser.add_argument('--condition', type=str, default = 'comprehension',
                        help='comprehension mode for caregivers; production mode for children output')
    
    parser.add_argument('--threshold_range', type=list, default = [600],
                        help='threshold to decide knowing a productive word or not, one of the variable to invetigate')
    
    parser.add_argument('--eval_path', type=str, default = 'Output/BE/production/cleaned_CDI/Oxford_CDI',
                        help='path to the evaluation material; one of the variables to invetigate')
    
    parser.add_argument('--freq_type', type=str, default = 'cum',
                        help='cumulative freq(cum) or raw fequency(raw)')
    
    parser.add_argument('--mode', type=str, default = 'month',
                        help='different chunk modes: by fixed chunks or by month')
    
    return parser.parse_args(argv)



def load_data(TextPath,output_dir,file,lang,condition):
    
    '''
    load and data in the corpus 
    input: data dir for one file
    output: 1. the cleaned transcript: lang/month
            2. a dataframe with filename|month|sent num|token count|token type count
    '''
    
    CHApath = TextPath + '/' + lang + '/' + file 
    
    sent, word = clean_text(CHApath,condition)
    
    # flatten the content and word list
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    
    file_dir = output_dir + '/' + file.split('.')[0] + '.txt'
    with open(file_dir, 'w', encoding="utf8") as f:
        for line in sent:
            f.write(line + '\n') 
            
    
    month = file.split('_')[-1].split('.')[0]
    genre = file.split('_')[0]
    speaker = file.split('_')[1]
    
    file_stat = pd.DataFrame([file, int(month), lang, genre, speaker, file_dir, len(sent), len(word), len(set(word))]).T
    file_stat = file_stat.rename(columns={0: "filename", 1: "month", 2: "lang" , 3: "genre" , 4:"child",5:'path',
                                          6:'sent_num', 7:'token_num', 8:'token_type_num'})
   
    return file_stat
    

   
def count_by_month(OutputPath,file_stat_sorted):
    
    '''
    aggregate by month
    '''
    
    month_lst = list(set(file_stat_sorted['month'].tolist()))
    group_stat = pd.DataFrame()
    
    for end_month in month_lst:
        
        start_month = end_month - 1
        
        selected_stat = file_stat_sorted[file_stat_sorted['month'] == end_month]
        
        # read and concatenate the transcripts
        text_lst = []
        word = []
        for path in selected_stat['path'].tolist():
            with open(path, encoding="utf8") as f:
                file = f.readlines()
                text_lst.append(file)
            
        # flatten and write the concatenated transcripts
        text_concat = [item for sublist in text_lst for item in sublist]
        
        # print out the text files

        with open(OutputPath + '/Transcript_by_month/transcript_' + str(end_month) + '.txt', 'w', encoding="utf8") as f:
            for line in text_concat:
                f.write(line + '\n')  
        
        
        # get the number of tokens
        word_lst = ' '.join(text_concat)
        word = word_lst.split(' ')
        
        # path and filenames are connected by "," respectively
        paths = ','.join(selected_stat['path'].tolist())
        filenames = ','.join(selected_stat['filename'].tolist())
        
        # get the age span in month
        # get the stat for the con catenated transcripts: token/sent num, token type num, file paths, filenames
        group = str(start_month) + '-' + str(end_month)
        file_stat = pd.DataFrame([group, paths, filenames, len(text_concat), len(word), len(set(word)),start_month,end_month]).T
        file_stat = file_stat.rename(columns={0: "group", 1: "paths", 2: "files", 3:'sent_num', 4:'token_num', 5:'token_type_num',6:'start_month', 7:'end_month'})
        group_stat = pd.concat([group_stat,file_stat])  
        
    
    # print out thet stat frame
    if not os.path.exists(OutputPath + '/month'):
        os.makedirs(OutputPath + '/month') 
    
    # add cumulative one
    group_stat['token_num_cum'] = group_stat['token_num'].cumsum()
    group_stat['hypothesized'] = group_stat['end_month'] * 30 * 10000
     
    group_stat['weight_factor'] = group_stat['hypothesized'] / group_stat['token_num_cum']
    group_stat.to_csv(OutputPath + '/month/Stat_chunk.csv')
    return group_stat
  



def get_chunks(OutputPath,file_stat_sorted,Chunksize):
    
    word_chunk = Chunksize * 60 * 180
    # cumulative sums of the token numbers
    file_stat_sorted['Cumu_token_num'] = file_stat_sorted['token_num'].cumsum()
    
    # prune the chunk that is less than standard
    chunks = chunk_list(file_stat_sorted['Cumu_token_num'].tolist(),word_chunk)
    print('Get {} chunks altogether!'.format(str(len(chunks))))
    # concatenate transcripts and get the meta dataframe
    group_stat = pd.DataFrame()
    n = 0
    
    for chunk in chunks:
        
        selected_stat = file_stat_sorted[file_stat_sorted['Cumu_token_num'].isin(chunk)]
        
        # read and concatenate the transcripts
        text_lst = []
        word = []
        for path in selected_stat['path'].tolist():
            with open(path, encoding="utf8") as f:
                file = f.readlines()
                text_lst.append(file)
            
        # flatten and write the concatenated transcripts
        text_concat = [item for sublist in text_lst for item in sublist]
        
        group = str(Chunksize) + '_' + str(n)
        if not os.path.exists(OutputPath + '/' + str(Chunksize)):
            os.makedirs(OutputPath + '/' + str(Chunksize)) 
            
        with open(OutputPath + '/' + str(Chunksize) + '/' + group + '.txt', 'w', encoding="utf8") as f:
            for line in text_concat:
                f.write(line + '\n')  
                result = line.split(' ')
                for j in result:
                    cleaned_word = j.strip()
                    if len(cleaned_word) > 0:  
                        word.append(cleaned_word)
    
        # path and filenames are connected by "," respectively
        paths = ','.join(selected_stat['path'].tolist())
        filenames = ','.join(selected_stat['filename'].tolist())
        
        max_month = selected_stat['month'].max()
        min_month = selected_stat['month'].min()
        
        # get the age span in month
        # get the stat for the con catenated transcripts: token/sent num, token type num, file paths, filenames
        file_stat = pd.DataFrame([group, paths, filenames, len(text_concat), len(word), len(set(word)),min_month,max_month]).T
        # add the theoretical number in the corresponding month and cumu token numbers
        file_stat = file_stat.rename(columns={0: "group", 1: "paths", 2: "files", 3:'sent_num', 4:'token_num', 5:'token_type_num',6:'start_month', 7:'end_month'})
        group_stat = pd.concat([group_stat,file_stat])         
        
        n += 1
        
    group_stat['hypothesized'] = group_stat['end_month'] * 30 * 10000
    group_stat['weight_factor'] = group_stat['hypothesized'] / group_stat['token_num']
    
    # print out thet stat frame
    group_stat.to_csv(OutputPath + '/' + str(Chunksize) + '/Stat_chunk.csv')
    return group_stat
  


def count_words(OutputPath,Chunksize, group_stat,eval_path, threshold,freq_type,mode):
    
    '''
    count words in each chunk or month, get the cumulative count afterwards
    input: word counts in each chunk/month and the stat of the number of true words as well as the true data proportion
    
    output: a dataframe with each word count        
    the chunks are cumulative as they are first ranked and grouped
    '''
    # note we use cumulative tken numbers as well to get the learning curve
    # group_stat['weight_factor'] =  group_stat['hypothesized']/ group_stat['token_num']
    
    if eval_path.split('/')[-1].split('_')[0] == 'intersection':
        eval_lst = pd.read_csv(eval_path + '/WS.csv')['words'].tolist()
    else:
        eval_lst = pd.read_csv(eval_path + '/Freq_selected_content.csv')['words'].tolist()
        
    freq_frame = pd.DataFrame()
    freq_frame['word'] = eval_lst
    
    score_frame = pd.DataFrame()
    score_frame['word'] = eval_lst
    
    threshold_frame = pd.DataFrame()
    threshold_frame['word'] = eval_lst
    
    
    for file in group_stat['group'].tolist():
        
        # get word freq list for each file
        if mode == 'chunk':
            file_path = OutputPath + '/' + str(Chunksize) + '/' + file + '.txt'
        else: 
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
                count = fre_table[fre_table['Word']==word]['Freq'].item()
                
            except:
                count = 0
            freq_lst.append(count)
        freq_frame[file] = freq_lst
        
        # get cum freq 
    if freq_type == 'cum':
            
        # Get column names       
        sel_frame = freq_frame.iloc[:,1:]
        columns = freq_frame.columns[1:]
        sel_frame = sel_frame.cumsum(axis=1)
            
        for col in columns.tolist():
            freq_frame[col] = sel_frame[col] 
    
    # get each chunk's scores based on the threshold
    columns = freq_frame.columns[1:]
      
    for col in columns.tolist():
        
        score_lst = []
        #thre = group_stat[group_stat['group'] == col]['start_month'].item() * threshold
        
        for count in freq_frame[col].tolist():
            
            # normalize the count for the given month
            norm_count = group_stat[group_stat['group'] == col]['weight_factor'].item() * count
            
            # varying thresholds based on the correction factors
            if norm_count >= threshold:
                score = 1
            else:
                score = 0
                        
            score_lst.append(score)
                
        score_frame[col] = score_lst
    
    
    if not os.path.exists(eval_path):
        os.makedirs(eval_path) 
        
    freq_frame.to_csv(eval_path + '/' + freq_type + '_selected_freq.csv')
    
    score_frame.to_csv(eval_path + '/'+ freq_type +'_score_' + str(threshold) +'.csv')
    return freq_frame, score_frame
    




def plot_bar(eval_path, group_stat,Chunksize, threshold,freq_type):
    score_frame = pd.read_csv(eval_path + '/'+ freq_type +'_score_' + str(threshold) +'.csv')
    column_list = score_frame.columns.tolist()[2:]
    line_x = []
    for chunk in column_list: 
        min_month = group_stat[group_stat['group'] == chunk]['start_month'].item()
        max_month = group_stat[group_stat['group'] == chunk]['end_month'].item()
        column_name = str(min_month) + '-' + str(max_month)
        score_frame.rename(columns={chunk: column_name}, inplace=True)     
        line_x.append(min_month)
    # get the average-frame 
    column_averages = score_frame.mean()[1:]
    
    # Columns to plot
    columns_to_plot = column_averages.index
    
    # Customized bar widths
    column_widths = []
    bar_positions = []
    labels = []
    label_positions = []
    
    for column in columns_to_plot:
        column_left = column.split('-')[0]
        column_right = column.split('-')[1]
        width = 0.1 * (int(column_right) - int(column_left))
        bar_pos = sum(column_widths) + width/2
        label_pos = sum(column_widths)
        column_widths.append(width)
        bar_positions.append(bar_pos)
        
        labels.append(column_left)
        label_positions.append(label_pos)
    
    # add the last one on the right
    labels.append(column_right)
    label_positions.append(sum(column_widths))
    
    # Create bar chart with custom widths
    n = 0
    while n < len(columns_to_plot):
        
        plt.bar(bar_positions[n], column_averages[n], width=column_widths[n], align='center') 
        n += 1
    
        
    plt.xticks(label_positions, labels)
    plt.xlabel('months')
    plt.ylabel('Proportion of kids')
    plt.title('CHILDES expressive vocab (threshold = {}; chunksize = {})'.format(threshold,Chunksize))
    plt.legend()
    
    figure_path = eval_path + '/Figure_backup/'
    if not os.path.exists(figure_path):
        os.makedirs(figure_path) 
        
    plt.savefig(figure_path + '/Bar_' + str(Chunksize) + '_' + str(threshold) + '.png')
    plt.show()

  


def calculate_fitness(mean_values, mean_value_CDI):
    
    '''
    input: series of datapoints of two curves with month as the column index
    
    output: the fitness scores of the two curves 
    ''' 

    # resample the shorter month range so that the curve month range to be compared are aligned
    # Get overlapping index values
    common_index = mean_values.index.intersection(mean_value_CDI.index)
    
    # Sample series1 based on common index
    series1 = mean_values.loc[common_index]   
    # Sample series2 based on common index
    series2 = mean_value_CDI.loc[common_index]
    
    if len(series1) != len(series2):
        # Interpolate the curves to have the same number of data points
        length = max(len(series1), len(series2))
        x = np.linspace(0, 1, length)
        f1 = interp1d(np.linspace(0, 1, len(series1)), series1.tolist())
        f2 = interp1d(np.linspace(0, 1, len(series2)), series2.tolist())

        curve1 = f1(x)
        curve2 = f2(x)
    else:
        # Convert curves to numpy arrays for easier computation
        curve1 = np.array(series1.tolist())
        curve2 = np.array(series2.tolist())

    # Calculate the root mean square error (RMSE)
    rmse = np.sqrt(np.mean((curve1 - curve2) ** 2))

    # Invert the RMSE value to obtain a fitness score
    fitness = 1.0 / (1.0 + rmse)

    return fitness

'''
eval_path = 'Output/AE/production/cleaned_CDI/WS'
Chunksize = 0.5
threshold = 100
freq_type = 'cum'
OutputPath = 'Output/AE'
group_stat = pd.read_csv(OutputPath + '/production/month/Stat_chunk.csv')
'''


def compare_curve(eval_path,Chunksize,threshold,freq_type,group_stat):
    
    sns.set_style('whitegrid') 
    score_frame = pd.read_csv(eval_path + '/'+ freq_type +'_score_' + str(threshold) +'.csv')
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
    option1: get the average value of the same bar/month
    '''
    mean_values = df.groupby('month')['Value'].mean()
    plt.plot(mean_values, label='CHILDES')
    
    '''
    option2: get the max value of the same bar
    '''
    # plot the CDI results
    selected_words = pd.read_csv(eval_path + '/Freq_selected_content.csv').iloc[:, 5:-4]
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
        
    plt.title('CHILDES expressive vocab (threshold = {})'.format(threshold))
    plt.xlabel('age in month', fontsize=15)
    plt.ylabel('Proportion of children', fontsize=15)

    plt.tick_params(axis='both', labelsize=10)
  
    plt.legend()
    figure_path = eval_path + '/Figure_backup/'
    if not os.path.exists(figure_path):
        os.makedirs(figure_path) 
    plt.savefig(figure_path + '/Curve_' + str(Chunksize) + '_' + str(threshold) + '.png')
    plt.show()
    
    
    mean_value_CDI = data_frame_final.groupby('month')['Proportion of acquired words'].mean()
    fitness = calculate_fitness(mean_values, mean_value_CDI)
    return fitness
      

def plot_trend(result_frame):
   
    # Create a scatter plot with color mapping
    plt.scatter(result_frame['Chunksize'].tolist(), result_frame['threshold'].tolist(), c=result_frame['rmse'].tolist(), cmap='viridis')
    
    # Set labels and title
    plt.xlabel("Chunksize")
    plt.ylabel('threshold')
    plt.title('Fitness of estimated vocabulary size')

    # Add a colorbar for reference
    plt.colorbar(label='rmse')

    # Show the plot
    plt.show()


def plot_threshold(result_frame, eval_path):
    eval_path = 'Output/BE/production/cleaned_CDI/Oxford_CDI'
    result_frame = pd.read_csv(eval_path + '/' + 'Fitness_All.csv')
    result_frame = result_frame[result_frame['Chunksize'] == 0.5]
    ax = sns.lineplot(x="threshold", y="rmse", data=result_frame)
    for line in ax.lines:
        plt.ylim(0.5,1)
    # Set labels and title
    plt.xlabel("threshold")
    plt.ylabel('rmse')
    plt.title('Estimated expressive vocabulary size')

    # Show the plot
    plt.show()



def main(argv):
    
    # Args parser
    args = parseArgs(argv)
    
    mode = args.mode
    freq_type = args.freq_type
    TextPath = args.TextPath
    condition = args.condition
    eval_path = args.eval_path
    OutputPath = args.OutputPath + '/' + condition

    Chunksize_range = args.Chunksize_range
    threshold_range = args.threshold_range
    
    result_frame = pd.DataFrame()
    log_frame = pd.DataFrame()
    for Chunksize in Chunksize_range:
        for threshold in threshold_range:
    
            # step 1: load and clean transcripts
            if not os.path.exists(OutputPath + '/stat_per_file.csv'):   
            #if os.path.exists(OutputPath + '/stat_per_file.csv'):   
                print('Start cleaning transcripts')
                
                file_stat = pd.DataFrame()
                for lang in os.listdir(TextPath):  
                    output_dir =  OutputPath + '/' + lang
                    for file in os.listdir(TextPath + '/' + lang): 
                        
                        file_frame = load_data(TextPath,output_dir,file,lang,condition)
                        file_stat = pd.concat([file_stat,file_frame])
               
                file_stat_sorted = file_stat.sort_values('month')    
                file_stat_sorted.to_csv(OutputPath + '/stat_per_file.csv')        
                        
                print('Finished cleaning files')
                
            else:    
                print('The file already exists! Skip')
                file_stat_sorted = pd.read_csv(OutputPath + '/stat_per_file.csv')
                
            # step 2: cocnatenate transcripts
            if mode == 'chunk':
                chunk_path = OutputPath + '/' + str(Chunksize) + '/Stat_chunk.csv'
                
            elif mode == 'month':
                chunk_path = OutputPath + '/month/Stat_chunk.csv'
                
            if not os.path.exists(chunk_path):   
            #if os.path.exists(chunk_path):     
                
                if mode == 'chunk':
                    print('Starting building chunks with the size of {}'.format(str(Chunksize)))
                    group_stat = get_chunks(OutputPath,file_stat_sorted,Chunksize)
                    
                elif mode == 'month':
                    print('Starting counting by month')
                    group_stat = count_by_month(OutputPath,file_stat_sorted)
                    
            else:    
                print('The concatenated chunks already exist! Skip')
                group_stat = pd.read_csv(chunk_path)
            
              
            # step 3: get word freq in each chunk(go through this anyway) 
            score_path = eval_path + '/'+ freq_type +'_score_' + str(threshold) +'.csv'
               
            print('Starting counting selected evluation words')
            freq_frame, score_frame = count_words(OutputPath,Chunksize, group_stat,eval_path, threshold,freq_type,mode)
            print('Finished counting selected evluation words')
            
            
    
            try:
                # step 4: plot the developmental bars
                print('Plotting the developmental bars')
                plot_bar(eval_path, group_stat,Chunksize, threshold,freq_type)
                
                # compare and get the best combination of threshold two variables
                rmse = compare_curve(eval_path,Chunksize,threshold,freq_type,group_stat)
                
                temp_frame = pd.DataFrame([Chunksize, threshold, rmse]).T
                
                file_stat = temp_frame.rename(columns={0: "Chunksize", 1: "threshold", 2: "rmse" })
                
                fitness_dir = eval_path + '/Fitness_backup/'
                if not os.path.exists(fitness_dir):
                    os.makedirs(fitness_dir) 
                    
                file_stat.to_csv(fitness_dir + str(Chunksize) + '_' + str(threshold) + '.csv')
                # save the results in a dataframe
                result_frame = pd.concat([result_frame,file_stat])
                
            except:
                temp_log = pd.DataFrame([Chunksize, threshold]).T
                log_stat = temp_log.rename(columns={0: "Chunksize", 1: "threshold" })
                log_frame = pd.concat([log_frame,log_stat])
                print([Chunksize, threshold])
    
    
    # step 5: plot the developmental bars
    result_frame.to_csv(eval_path + '/Fitness_All.csv')  
    log_frame.to_csv(eval_path + '/Log_All.csv')  
    plot_trend(result_frame)
    
    
    

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    
    
 



