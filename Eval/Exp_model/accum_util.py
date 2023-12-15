# -*- coding: utf-8 -*-
"""
plot the accumulator result
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


"""
Step1: put all the results in one csv file

/scratch1/projects/InfTrain/experiences/swuggy_hadrien/lexical/en/testset_64

output: freq and avg score
"""

# aggregae columns by freq bads

def get_freq(freq,num_cols,interval,name):
    
    i = 0
    while i < num_cols:
        
        freq[name+'_' + str(int(i/interval))] = freq.iloc[:, i:i+interval].sum(axis=1)
        i += interval
    return freq





def group_freq(lang,file_dir):
    configs = [[2,'100h'],[4,'200h'],[8,'400h'],[16,'800h'],[32,'1600h'],[64,'3200h']]
    # READ THE DATA 
    # freq = pd.read_csv('freq_'+lang +'_all.csv')
    # freq =  freq.iloc[:, 1:]
    freq = pd.read_csv(file_dir)
    freq =  freq.iloc[:, 1:]
    group_lst = freq['group'].tolist()
    freq =  freq.iloc[:, 19:]
    num_cols = freq.shape[1]
    for config in configs: 
        freq = get_freq(freq,num_cols,config[0],config[1])
    return freq, group_lst


def get_score(freq,threshold):
    final_score = []
    final_name = []
    final_month = []
    
    # loop through each column
    for col_name, col_data in freq.iteritems():
        score_lst = []
        for i in col_data.values.tolist():
            if i >= threshold:
                score = 1
            else:
                score = 0
            score_lst.append(score) 
            # average scores among different words
            # add the freq bin here! 
            
            mean_score = sum(score_lst)/len(score_lst)
        final_score.append(mean_score)
        # rename the score 
        if col_name.endswith('.txt'):
            name = '50'
        else:
            name = col_name.split('_')[0][:-1]
        month = int(name)/89
        final_name.append(name)
        final_month.append(month)
    
    # convert into datframe
    accum_frame = pd.DataFrame([final_name,final_month,final_score]).T
    
    headers =['Quantity of speech (h)', 'month', 'Lexical score']
    accum_frame.columns = headers
    accum_frame['threshold'] = threshold
    return accum_frame



def plot_accum_threshold(freq,threshold_lst,lang):
    
    accum_frame = pd.DataFrame()
    for threshold in threshold_lst:
        single_threshold = get_score(freq,threshold)
        accum_frame = pd.concat([accum_frame, single_threshold]) 
    # plot the curve
    sns.set_style('whitegrid')
    sns.lineplot(data=accum_frame, x="month", y='Lexical score',hue='threshold')
    if lang == 'EN':
        title_name = 'English model'
    elif lang == 'FR':
        title_name = 'French model'
    plt.title(title_name)
    plt.legend(bbox_to_anchor=(1.05, 1))
    # Show the plot
    plt.savefig('Accum_' + lang +'.png')
    plt.show()  
    


def get_score_freq(freq,threshold,freq_bin,group_lst):

    # plot the results based on different freq bins
    final_score = []
    final_name = []
    final_month = []
    # select rows with given freq
    freq['group'] = group_lst
    freq_selected = freq[freq['group'] == freq_bin]
    freq_selected.pop(freq_selected.columns[-1])
    # loop through each column
    for col_name, col_data in freq_selected.iteritems():
        score_lst = []
        for i in col_data.values.tolist():
            if i >= threshold:
                score = 1
            else:
                score = 0
            score_lst.append(score) 
            # average scores among different words in the given freq bins
             
            mean_score = sum(score_lst)/len(score_lst)
        final_score.append(mean_score)
        # rename the score 
        if col_name.endswith('.txt'):
            name = '50'
        else:
            name = col_name.split('_')[0][:-1]
        month = int(name)/89
        final_name.append(name)
        final_month.append(month)
    
    # convert into datframe
    accum_frame = pd.DataFrame([final_name,final_month,final_score]).T
    
    headers =['Quantity of speech (h)', 'month', 'Lexical score']
    accum_frame.columns = headers
    accum_frame['freq_bin'] = freq_bin
    return accum_frame


 
def plot_accum_freq(freq,freq_bands,threshold,lang):
    
    accum_frame = pd.DataFrame()
    for freq_bin in freq_bands:
        single_threshold = get_score_freq(freq,threshold,freq_bin)
        accum_frame = pd.concat([accum_frame, single_threshold]) 
    # plot the curve
    sns.set_style('whitegrid')
    sns.lineplot(data=accum_frame, x="month", y='Lexical score',hue='freq_bin')
    if lang == 'EN':
        title_name = 'Accumulator model'
    elif lang == 'FR':
        title_name = 'French model'
    plt.title(title_name, fontsize=20)
    
    plt.xlabel('pseudo age in month', fontsize=18)
    plt.ylabel('Proportion of words', fontsize=18) 
    
    plt.tick_params(axis='both', labelsize=13) 

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=15, title='Freq band',title_fontsize=18)
    
    # Show the plot
    plt.savefig('Accum_' + lang +'.png')
    plt.show()  

    
    


def aggregate_frame(threshold):    
    file_dir = 'STELA/gold_raw_filtered.csv'    
    lang = 'EN'   
    freq_bands = ['low[5,140]', 'high[140,3083]']  
    freq, group_lst = group_freq(lang,file_dir)
    
    accum_frame = pd.DataFrame()
    for freq_bin in freq_bands:
        single_threshold = get_score_freq(freq,threshold,freq_bin)
        accum_frame = pd.concat([accum_frame, single_threshold]) 
    # plot the curve
    sns.set_style('whitegrid')
    order=['low[5,140]', 'high[140,3083]']
    sns.lineplot(x="month", y="Lexical score", data=accum_frame, hue="freq_bin",palette=['b','r'], hue_order=order)
    
    plt.title('Accumulator model', fontsize=20)
    
    plt.xlabel('pseudo age in month', fontsize=18)
    plt.ylabel('Proportion of words', fontsize=18) 
    
    plt.tick_params(axis='both', labelsize=13) 
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=15, title='Freq band',title_fontsize=18)
    
    # Show the plot
    plt.savefig('Accum_' + lang +'.png')
    plt.show()  
    
    # aggregate the results: for each chunk and each freq band, we have the range of the result
    whole_frame = pd.DataFrame()
    hour_lst = ['50','100','200','400','800','1600','3200']
    freq_lst=['low[5,140]', 'high[140,3083]']
    for hour in hour_lst:
        accum_frame_hour = accum_frame[accum_frame['Quantity of speech (h)'] == hour]
        for freq in freq_lst:
            accum_frame_freq = accum_frame_hour[accum_frame_hour['freq_bin'] == freq]
            # sort the frame
            sorted_accum_frame_freq = accum_frame_freq.sort_values(by='Lexical score')
            max_frame = sorted_accum_frame_freq.head(1)
            min_frame = sorted_accum_frame_freq.tail(1)
            range_frame = pd.concat([max_frame,min_frame])
            whole_frame = pd.concat([whole_frame,range_frame])
            
    whole_frame.to_csv('agg_accum_2_' + str(threshold)+'.csv')
    accum_frame.to_csv('accum_2_' + str(threshold)+'.csv')
    # aggregate by each chunk(get range of the results)


