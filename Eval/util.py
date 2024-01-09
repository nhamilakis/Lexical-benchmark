import string
import pandas as pd
import re
import collections
import math
import enchant
import os
import numpy as np

# create dictionary for the language
# in use(en_US here)
d = enchant.Dict("en_US")



def load_transcript(TextPath,output_dir,file,lang,condition): 
    
    '''
    load data in the corpus 
    input: data dir for one file
    output: 1. the cleaned transcript: lang/month
            2. a dataframe with filename|month|sent num|token count|token type count
    '''
    
    
    def clean_text(path,condition):
        
        '''
        clean each .cha transcript
        input: 1. path to the .cha transcripts from CHILDES
               2. receotive or expressive vocab(related with the speaker)
        ouput: [the cleaned transcript],[the word list of the cleaned transcript]
        '''
        with open(path, encoding="utf8") as f:
            file = f.readlines()
            cleaned_lst = []
            for script in file: 
                if condition == 'comprehension':
                    if script.startswith('*') and not script.startswith('*CHI'):
                        # remove annotations
                        script = re.sub('\[[=\?@#%^&*!,.()*&_:;\'\"]*\s*\w*\]', '', script)
                        script = re.sub('[&=]+\s*\w+', '', script)
                        script = re.sub('\[!=+\s*\w+(\s*\w*)*]', '', script)
                        script = re.sub('&=[A-Za-z]+', '', script)
                        
                        translator = str.maketrans('', '', string.punctuation+ string.digits)
                        clean_string = script.translate(translator).lower()
                        cleaned_lst.append(clean_string)
                else:
                    if script.startswith('*') and script.startswith('*CHI'):
                        # remove annotations
                        script = re.sub('\[[=\?@#%^&*!,.()*&_:;\'\"]*\s*\w*\]', '', script)
                        script = re.sub('[&=]+\s*\w+', '', script)
                        script = re.sub('[!=]+\s*\w+', '', script)
                        # remove punctuations and digits
                        translator = str.maketrans('', '', string.punctuation+ string.digits)
                        clean_string = script.translate(translator).lower()
                        cleaned = clean_string.replace('\n','')
                        cleaned_lst.append(cleaned)    
                       
        trial = pd.DataFrame(cleaned_lst, columns = ['whole'])
        splitted = trial['whole'].str.split('\t', expand=True)
        splitted.rename(columns = {0:'speaker', 1:'content'}, inplace = True)
        
        
        if splitted.shape[0] > 0:   
            trial_lst = splitted['content'].tolist()
            result_lst = []
            cleaned_text = []
            for i in trial_lst:
                removed_start = i.lstrip()
                removed_end = removed_start.rstrip() 
                removed_multi = re.sub(r"\s+", " ", removed_end)
                
                if len(removed_multi) > 0:
                    cleaned_text.append(removed_multi)
                    
                    result = removed_multi.split(' ')
                    
                    for j in result:
                        cleaned_word = j.strip()
                        if len(cleaned_word) > 0:  
                            result_lst.append(cleaned_word)
       
            return cleaned_text, result_lst
        else:
            return [],[]
    
    
    
    
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
    
    file_stat = pd.DataFrame([file, month, lang, genre, speaker, file_dir, len(sent), len(word), len(set(word))]).T
    file_stat = file_stat.rename(columns={0: "filename", 1: "month", 2: "lang" , 3: "genre" , 4:"child",5:'path',
                                          6:'sent_num', 7:'token_num', 8:'token_type_num'})
   
    return file_stat


def count_by_month(OutputPath,file_stat_sorted):
    
    '''
    concatenate word info in each month
    
    input: the output path for each month's input'
           output path to store such info
    output: csv file with all file info by month
    '''
    
    month_lst = list(set(file_stat_sorted['month'].tolist()))
    group_stat = pd.DataFrame()
    
    transcript_dir = OutputPath + '/Transcript_by_month'
        
    if not os.path.exists(transcript_dir):
        os.makedirs(transcript_dir) 
   
    
    for end_month in month_lst:
        
        start_month = int(end_month) - 1
        
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
        
        
            
        # get the concatenated transcripts by month
        with open(transcript_dir + '/transcript_' + str(end_month) + '.txt', 'w', encoding="utf8") as f:
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
    
    group_stat.to_csv(OutputPath + '/month/Stat_chunk.csv')
    return group_stat
  
def get_score(freq_frame,OutputPath,threshold,hour):
    
    '''
    get scores of the target words in Wordbank 
    input: word counts in each chunk/month and the stat of the number of true words as well as the true data proportion
    output: a dataframe with each word count  

    we have the weighed score in case the influence of different proportions of frequency bands      
    '''
     
    
    freq_frame = freq_frame.drop(columns=['word', 'group_original'])
    
    # get each chunk's scores based on the threshold
    columns = freq_frame.columns
      
    for col in columns.tolist():
        freq_frame.loc[col] = [0] * freq_frame.shape[1]
        
    # get the score based on theshold
    score_frame = freq_frame.applymap(lambda x: 1 if x >= threshold else 0)
    
    avg_values = score_frame.mean()
    
    score_path = OutputPath + '/Scores/'
    if not os.path.exists(score_path):
        os.makedirs(score_path) 
    score_frame.to_csv(score_path + '/score_' + str(threshold) +'.csv')
    
    return score_frame, avg_values
    


def get_freq(result):
    
    '''
    input: raw word list extracted from the transcripts 
    output: the freq dataframe with all the words and their raw freq
    '''
    
    frequencyDict = collections.Counter(result)  
    freq_lst = list(frequencyDict.values())
    word_lst = list(frequencyDict.keys())
    
    # get freq
    fre_table = pd.DataFrame([word_lst,freq_lst]).T
    col_Names=["Word", "Freq"]
    fre_table.columns = col_Names
    fre_table['Norm_freq'] = fre_table['Freq']/len(result)
    fre_table['Norm_freq_per_million'] = fre_table['Norm_freq']*1000000
    # get log_freq
    log_freq_lst = []
    for freq in freq_lst:
        log_freq = math.log10(freq)
        log_freq_lst.append(log_freq)
    fre_table['Log_freq'] = log_freq_lst
    
    # get logarithm of normalized word freq per million
    norm_log_freq_lst = []
    for freq in fre_table['Norm_freq_per_million'].tolist():
        norm_log_freq = math.log10(freq)
        norm_log_freq_lst.append(norm_log_freq)
    fre_table['Log_norm_freq_per_million'] = norm_log_freq_lst
    
    return fre_table





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




def plot_bar(eval_path, group_stat,threshold):
    
    '''
    plot each months' average score based on the score path'
    '''
    
    
    score_frame = pd.read_csv(eval_path + '/score_' + str(threshold) +'.csv')
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
    plt.title('CHILDES expressive vocab (threshold = {})'.format(threshold))
    plt.legend()
    
    figure_path = eval_path + '/Figures/'
    if not os.path.exists(figure_path):
        os.makedirs(figure_path) 
        
    plt.savefig(figure_path + '/Bar_' + str(threshold) + '.png')
    plt.show()

  

def plot_trend(result_frame):
   
    # Create a scatter plot with color mapping
    plt.scatter(result_frame['threshold'].tolist(), c=result_frame['rmse'].tolist(), cmap='viridis')
    
    # Set labels and title
    plt.xlabel("Chunksize")
    plt.ylabel('threshold')
    plt.title('Fitness of estimated vocabulary size')

    # Add a colorbar for reference
    plt.colorbar(label='rmse')

    # Show the plot
    plt.show()
