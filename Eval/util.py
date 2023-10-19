import string
import pandas as pd
import re
import numpy as np
import collections
import math
import os
import enchant
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt



# create dictionary for the language
# in use(en_US here)
d = enchant.Dict("en_US")

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


def get_prompt(path): 
    
    '''
    get the conversation pairs from each .cha transcript
    input: path to the .cha transcripts from CHILDES     
    ouput: a dataframe of the cleaned .CHA file with segemnted conversation pairs 
    '''
    
    def find_consec(sorted_list):
        consecutive_groups = []
        current_group = [sorted_list[0]]
    
        for i in range(1, len(sorted_list)):
            if sorted_list[i] == sorted_list[i-1] + 1:
                current_group.append(sorted_list[i])
            else:
                consecutive_groups.append(current_group)
                current_group = [sorted_list[i]]
    
        consecutive_groups.append(current_group)
        return consecutive_groups
    
    def merge_consec(splitted,index):
        
        '''
        merge the lists
        input: the splitted frame
        output: dataframe [earliest index,transcript]
        '''
        
        text_all = []
        index_all = []
        for index_lst in index:
            # go over the sublist
            text_rows = '. '.join(splitted.iloc[index_lst]['content'].tolist())
            text_all.append(text_rows)
            index_all.append(index_lst[0])
        # remap the columns 
        transcript = pd.DataFrame([index_all,text_all]).T
        # rename the dataframe
        transcript.rename(columns = {0:'index', 1:'text'}, inplace = True)
        return transcript

    with open(path, encoding="utf8") as f:
        file = f.readlines()
        cleaned_lst = []
        for script in file: 
            # remove annotations
            
            if script.startswith('*'):                
                    # clean the transcripts
                    script = re.sub('\[[=\?@#%^&*!,.()*&_:;\'\"]*\s*\w*\]', '', script)
                    script = re.sub('[&=]+\s*\w+', '', script)
                    script = re.sub('\[!=+\s*\w+(\s*\w*)*]', '', script)
                    script = re.sub('&=[A-Za-z]+', '', script)                
                    translator = str.maketrans('', '', string.punctuation+ string.digits)
                    clean_string = script.translate(translator).lower()
                    cleaned = clean_string.replace('\n','')
                    # remove annotations
                    substrings = ['xxx', 'yyy', 'noise', 'rustling', 'vocaliz', 'rattl','breath','sound','babbl','\x15',
                           'ow','cough','um','cry','eh','cries','moo', 'bang','coo','rasberr','transcrib','ha','sneez','bubbl','squeal']
                    
                    pattern = '|'.join(map(re.escape, substrings))
                    cleaned = re.sub(pattern, '', cleaned)
                    # detect if only the spaces left in this case
                    if cleaned.isspace():
                        pass 
                    else:
                        cleaned_lst.append(cleaned)    
                   
    trial = pd.DataFrame(cleaned_lst, columns = ['whole'])
    splitted = trial['whole'].str.split('\t', expand=True)
    splitted.rename(columns = {0:'speaker', 1:'content'}, inplace = True)

    # group into pairs
    child_lst = splitted[splitted['speaker'] == 'chi'].index.tolist()  #get row indices containing children's production
    # merge the consevutive utterances produced by children or parents
    parent_lst = splitted[splitted['speaker'] != 'chi'].index.tolist()
    
    child_index = find_consec(child_lst)
    parent_index =  find_consec(parent_lst)

    transcript_child = merge_consec(splitted,child_index)
    transcript_parent = merge_consec(splitted,parent_index)
    
    '''
    match two transcript dataframes
    parents' utterance shoudl precede the children's
    loop over the child's dataframe
    '''
    
    parent_pair = []
    n = 0
    while n < transcript_child.shape[0]: 
        
        # note that there might be multiple rows of the selected frame
        if n == 0:
            # no previous rows
            selected_rows = transcript_parent[transcript_parent['index'] < transcript_child['index'].tolist()[n]]

        else:
            # the selected transcript index should be between the previous and current rows
            selected_rows = transcript_parent[(transcript_parent['index'] > transcript_child['index'].tolist()[n-1]) & (transcript_parent['index'] < transcript_child['index'].tolist()[n])]
        
        # convert the frame into the transcript
        parent_text = '. '.join(selected_rows['text'].tolist())
        parent_pair.append(parent_text) 
        n += 1
        
    transcript_child['prompt'] = parent_pair   
    return transcript_child




def plot_curves(score_frame,group_stat,eval_path,threshold):
    
    '''
    plot curves of each equal-sized bins; map back to the month data
    input: word counts in each chunk
    output: 1. the cleaned transcript: lang/month
            2. a dataframe with filename|month|sent num|token count|token type count in the form of bar charts      
    '''
   
    # change the column name into the months spanned
    score_frame = pd.read_csv('Output/intersection_4/2/score.csv')
    group_stat = pd.read_csv('Output/EN/production/50/Stat_chunk.csv')
    column_list = score_frame.columns.tolist()[2:]
    for chunk in column_list: 
        min_month = group_stat[group_stat['group'] == chunk]['start_month'].item()
        max_month = group_stat[group_stat['group'] == chunk]['end_month'].item()
        column_name = str(min_month) + '-' + str(max_month)
        score_frame.rename(columns={chunk: column_name}, inplace=True)
        
    mode = 'production'
    word_type = 'content'
    group_type = 'word'
    lang = 'score'
    data_frame_final = get_data(score_frame, lang, mode, word_type,group_type)
    
    sns.set_style('whitegrid')
    ax = sns.lineplot(x="month", y="Proprotion of acquired words", data=data_frame_final, label='thrshold = 2')
    
        # set the limits of the x-axis for each line
    for line in ax.lines:
          #plt.xlim(0,36)
          plt.ylim(0,1)
    
    plt.xlabel('age in month', fontsize=18)
    plt.ylabel('Proportion of children', fontsize=18)
    
    plt.tick_params(axis='both', labelsize=14)
    
    
    if mode == 'comprehension':
            vocab = 'Receptive'
    elif mode == 'production':
            vocab = 'Productive'
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='upper left',fontsize=15, title='Type',title_fontsize=18)
    
    plt.title('Expressive vocabulary', fontsize=20)
        # display plot
    plt.savefig(lang + '_' + mode + '_developmental data.png')
    plt.show()


def get_freq(result):
    
    '''
    input: raw word list extracted from the transcripts 
    output: the freq dataframe with all the words adn their raw freq
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


def chunk_list(numbers,word_chunk):
    
    chunks = []
    current_chunk = []
    
    for number in numbers:
        if not current_chunk:
            current_chunk.append(number)
        elif number - current_chunk[0] <= word_chunk:
            current_chunk.append(number)
        else:
            chunks.append(current_chunk)
            current_chunk = [number]

    if current_chunk:
        chunks.append(current_chunk)
    
    # if it is less than 80% of the required chunks, we just dekete them
    if chunks[-1][-1] - chunks[-1][0] < word_chunk * 0.8:
        chunks = chunks[:-1]
    
    return chunks

def get_data(selected_words, lang, mode, word_type,group_type):
     
    size_lst = []
    month_lst = []
    group_lst = []
    n = 0
    while n < selected_words.shape[0]:

        if lang == 'Oxford':
            size_lst.append(selected_words.iloc[n].tolist()[6:19])
            group = selected_words[group_type].tolist()[n]
            group_lst.append([group] * 7)
            headers_list = selected_words.columns.tolist()[6:19]

        elif lang == 'WS_Short':
            size_lst.append(selected_words.iloc[n].tolist()[6:11])
            group = selected_words[group_type].tolist()[n]
            group_lst.append([group] * 5)
            headers_list = selected_words.columns.tolist()[6:11]

        elif lang == 'score':
            size_lst.append(selected_words.iloc[n].tolist()[2:])
            group = selected_words[group_type].tolist()[n]
            group_lst.append([group] * 9)
            headers_list = selected_words.columns.tolist()[2:]

        elif lang == 'TED':
            size_lst.append(selected_words.iloc[n].tolist()[6:21])
            group = selected_words[group_type].tolist()[n]
            group_lst.append([group] * 15)
            headers_list = selected_words.columns.tolist()[6:21]

        month_lst.append(headers_list)
        n += 1

    size_lst_final = [item for sublist in size_lst for item in sublist]
    month_lst_final = [item for sublist in month_lst for item in sublist]
    group_lst_final = [item for sublist in group_lst for item in sublist]
    month_lst_transformed = []
    for month in month_lst_final:
        month_lst_transformed.append(month)
    # convert into dataframe
    data_frame = pd.DataFrame([month_lst_transformed,size_lst_final,group_lst_final]).T
    data_frame.rename(columns={0:'month',1:'Proprotion of acquired words',2:'Freq band'}, inplace=True)
    data_frame_final = data_frame.dropna(axis=0)
    return data_frame_final

def is_word(lang, word_lst):
    
    '''
    input: a list of words, language 
    output: the result to check whether it is a word or not
    '''
    result_lst = []
    if lang == 'AE' or lang == 'BE':
        dictionary = enchant.Dict("en")
        
        
    for word in word_lst:
        
        # Check if the token is classified as a word
        result = dictionary.check(word)
        result_lst.append(result)
        
    return result_lst




def count_words(result,OutputPath,month,lang,condition):
    
    '''
    input: raw word list extracted from the transcripts 
    output: the freq dataframe with all the words adn their raw freq
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
    
    # check whether it is a word or not
    
    result_lst = []
    if lang == 'AE' or lang == 'BE':
        dictionary = enchant.Dict("en")
        
    for word in word_lst:
        
        # Check if the token is classified as a word
        try:
            result = dictionary.check(word)
        except: 
            result = False
        result_lst.append(result)

    
    fre_table['Result'] = result
    # print out the concatenated file
    output_dir = OutputPath + '/' + lang + '/' + condition + '/Freq_by_month'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    fre_table.to_csv(output_dir + '/Freq_'+ month +'.csv')
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






def get_AE_month(lang,condition): 
    
    
    '''
    output: a dataframe containing filenames, sent+num, word_num, word_type_num in each month
    
    100 words/minute
    
    constrcut similar amount of chcnks: 50h      
    ''' 
    
    # re-calculate the freq based on the updated transcripts
    transcript_path = 'Output/' + lang + '/' + condition + '/Transcript_by_month'
    stat_all = []
    for transcript in os.listdir(transcript_path):
        # read transcript as a word list
        word_list, sent_num, word_num = read_text(transcript_path + '/' + transcript)
        month = transcript.split('.')[0].split('_')[1]
        # output the freq list with an additional column to decide whether it si a word or not
        fre_table = count_words(word_list,OutputPath,month,lang,condition)
        word_type = fre_table[fre_table['Result']==True].shape[0]
        seq_type = fre_table.shape[0]
        stat_all.append([month,sent_num, word_num,seq_type,word_type]) 
        
    # convert  a nested list into a dataframe    
    df = pd.DataFrame(stat_all, columns=['month','sent_num','word_num','seq_type','word_type'])
    month_lst = []
    for m in df['month'].tolist():
            month_lst.append(int(m))
    df['month'] = month_lst
    
    df.to_csv('AE_production_stat.csv')
    df_sorted = df.sort_values('month')
    df_sorted['Cumu_sent_num'] = df_sorted['sent_num'].cumsum()
    df_sorted['Cumu_word_num'] = df_sorted['word_num'].cumsum()
    df_sorted['Cumu_seq_type'] = df_sorted['seq_type'].cumsum()
    return df



def chunk_list_median(lst,group_num):
    
    '''
    chunk list given the freq
    
    input: the freq list; the number of lists
    
    output: nested list containing each sublist
    '''
    
    n = len(lst)
    chunk_size = int(np.ceil(n/group_num))

    groups = []
    for i in range(0, n, chunk_size):
        group = lst[i:i+chunk_size]
        groups.append(group)

    while len(groups) > group_num:
        new_groups = []
        for i in range(0, len(groups)-1, 2):
            merged_group = groups[i] + groups[i+1]
            new_group_size = int(np.ceil(len(merged_group)/2))
            new_groups.extend([merged_group[:new_group_size], merged_group[new_group_size:]])
        if len(groups) % 2 != 0:
            new_groups.append(groups[-1])
        groups = new_groups

    medians = [np.median(group) for group in groups]
    median_diffs = [abs(medians[i+1] - medians[i]) for i in range(len(medians)-1)]
    while len(groups) < group_num:
        max_diff_index = median_diffs.index(max(median_diffs))
        split_group = groups[max_diff_index]
        new_group_size = int(np.ceil(len(split_group)/2))
        groups[max_diff_index:max_diff_index+1] = [split_group[:new_group_size], split_group[new_group_size:]]
        medians = [np.median(group) for group in groups]
        median_diffs = [abs(medians[i+1] - medians[i]) for i in range(len(medians)-1)]

    return groups


