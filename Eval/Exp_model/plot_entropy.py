#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script to compare entropy distr of the train data and genrated prompts

@author: jliu
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
from plot_entropy_util import mean_KL


root_path = 'eval'
gpu = False

def get_distr(root_path,gpu):
    
    '''
    get the entropy distr and KL divergence with the reference data for each type 
    
    input: the root directory containing all the generated files adn train reference
    output: 1.the info frame with all the generarted tokens
            2.the reference frame with an additional column of the month info
    
    '''
    
    
    # load the rerference data
    
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
                                        '''
                                        h_dist = mean_KL(data['LSTM_generated_h'].tolist(),train_distr['entropy'].tolist(),gpu)
                                        prob_dist = mean_KL(data['LSTM_generated_prob'].tolist(),train_distr['prob'].tolist(),gpu)
                                        
                                        h_dist_lst.append(h_dist)
                                        prob_dist_lst.append(prob_dist)
                                        '''
                                        h_dist_lst.append(1)
                                        prob_dist_lst.append(1)
                                        
                                        prob_all.append(data['LSTM_generated_prob'].tolist())
                                        h_all.append(data['LSTM_generated_h'].tolist())
                                        
                                        if strategy == 'beam':
                                            beam_lst.append(file.split('_')[0])
                                            topk_lst.append('0')
                                            topp_lst.append('0')
                                            random_lst.append('0')
                                            
                                        elif strategy == 'sample_topk':
                                            topk_lst.append(file.split('_')[0])
                                            beam_lst.append('0')
                                            topp_lst.append('0')
                                            random_lst.append('0')
                                            
                                        elif strategy == 'sample_topp':
                                            topp_lst.append(file.split('_')[0])
                                            beam_lst.append('0')
                                            topk_lst.append('0')
                                            random_lst.append('0')
                                            
                                        elif strategy == 'sample_random':
                                            random_lst.append('1')
                                            topk_lst.append('0')
                                            beam_lst.append('0')
                                            topp_lst.append('0')
                                            
                                            
                                            # concatnete all the basic info regarding the genrated seq
                                        strategy_lst.append(strategy.split('_')[1])
                                        prompt_lst.append(prompt_type)
                                        month_lst.append(month)
                                        temp_lst.append(float(file.split('_')[1]))
                                        directory_lst.append(month+ '/' + prompt_type + '/' + strategy + '/' + file)
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
    info_frame = info_frame[(info_frame['random'] != '.') & (info_frame['topp'] != '.')& (info_frame['topk'] != '.') & (info_frame['prob'] != '[]')]
    # remove additional lines if there is additional info
    
    # sort the result based on temperature to get more organized legend labels
    info_frame = info_frame.sort_values(by='temp', ascending=True)
    return info_frame, reference_frame




info_frame, reference_frame = get_distr(root_path,gpu)

info_all = pd.DataFrame()
decoding_lst = ['topk','beam','topp','random']
for decoding in decoding_lst:
    info_frame[info_frame]


    
def plot_single_para(reference,decoding,decoding_para,month,prompt,var):
    
    def plot_distr(reference_data, total_data,label_lst,x_label,title,prompt,month):
        
        '''
        plot the var dustribution     #!!! sort the labels
        input: 
            reference_data: a liost of the target training data
            tota_data: a list of the selected data in the given distr
            x_label: the chosen variable to investigate
            
        output: 
            the distr figure of the reference train set and the generated data 
        '''
        
        sns.set_style('whitegrid')
        sns.kdeplot(reference_data, fill=False, label='train set')
        
        n = 0
        while n < len(total_data):
        
            # Create the KDE plots without bars or data points
            ax = sns.kdeplot(total_data[n], fill=False, label=label_lst[n])
            
            n += 1
        
        if x_label == 'entropy':
            for line in ax.lines:
                plt.xlim(0,18)
                plt.ylim(0,1.5)
                
        else:
            for line in ax.lines:
                plt.xlim(0,19)
                plt.ylim(0,1)
                
        # Add labels and title
        plt.xlabel(x_label)
        plt.ylabel('Density')
        plt.title(prompt + ' generation, ' + title + ' in month ' + month)
        # Add legend
        plt.legend()
        # get high-quality figure
        plt.figure(figsize=(8, 6), dpi=800)
        plt.savefig(prompt + ' generation, ' + title + ' in month ' + month + '.png', dpi=800)
        # Show the plot
        plt.show()
        
    
    target = info_frame[(info_frame['month']==month) & (info_frame['decoding']==decoding) & (info_frame['prompt']==prompt)
                        & (info_frame[decoding]==decoding_para)]
    
    
    plot_distr(reference,target[var].tolist(),target['temp'].tolist(),var,decoding + ': ' + decoding_para,prompt,month)



def plot_distance(target,var,decoding,prompt,month):  
    
        
    # Create a scatter plot with color mapping
   
    plt.scatter(target['temp'].tolist(),target[decoding].tolist(),c=target[var + '_dist'].tolist(), cmap='viridis')
   
   # Set labels and title
    plt.xlabel("temp")
    plt.ylabel(decoding)
    plt.title(prompt + ' generation, ' + decoding + ' in month ' + month)

    # Add a colorbar for reference
    plt.colorbar(label='KL divergence')

    # Show the plot
    plt.show()
    








# loop different conditions to get multiple figures
# var_lst = ['entropy','prob']

var_lst = ['entropy','prob']
decoding_lst = ['topk','beam','topp','random']
prompt_lst = ['unprompted','prompted']

info_frame.to_csv('Info_frame.csv')
month_lst = ['1','3','12']


for month in month_lst:
    for var in var_lst:
        # load the reference data based on the month and the investigated variable
        reference = reference_frame[reference_frame['month']==month][var].tolist()
        for prompt in prompt_lst:
            
            for decoding in decoding_lst:
                
                
                # target = info_frame[(info_frame['month']==month) & (info_frame['decoding']==decoding) & (info_frame['prompt']==prompt)]
                
                # # sort the parameters 
                # decoding_val_lst = []
                # temp_val_lst = []
                # n = 0
                # while n < target.shape[0]:
                    
                #     decoding_val = int(target[decoding].tolist()[n])
                #     temp_val = float(target['temp'].tolist()[n])
                    
                #     decoding_val_lst.append(decoding_val)
                #     temp_val_lst.append(temp_val)
                #     n += 1
                
                # target[decoding] = decoding_val_lst
                # target['temp'] = temp_val_lst
                
                
                # # plot the KL divergence between the generated tokens and reference
                # plot_distance(target,var,decoding,prompt,month)
                
                
                # get the decoding-specific parameters
                decoding_para_lst = list(set(info_frame[(info_frame['month']==month) & (info_frame['decoding']==decoding) 
                                          & (info_frame['prompt']==prompt)][decoding].tolist()))
                for decoding_para in decoding_para_lst:
            
                    plot_single_para(reference,decoding,decoding_para,month,prompt,var)



                


    
 



    