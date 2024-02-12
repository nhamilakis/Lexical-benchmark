#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 22:36:20 2023
@author: jliu
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns


def load_CDI(human_frame):
    
    '''
    only load one freq band results
    '''
    start_idx = human_frame.columns.get_loc('category')
    end_idx = human_frame.columns.get_loc('word')
    human_result = human_frame.iloc[:, start_idx+1:end_idx]
    avg_values = human_result.mean()
    return human_result,avg_values



def load_accum(accum_all,accum_threshold):
    
    df = accum_all[accum_all['threshold']==accum_threshold]
    accum_result = df.groupby(['month'])['Lexical score'].mean().reset_index()
    
    return accum_result


    
def fit_sigmoid(x_data, y_data, target_y, offset,label,color,by_freq,style='solid'):
    '''
    fit sigmoid curve of extrapolated exp vocab

    '''

    def sigmoid(x, a, b):
        return 1 / (1 + np.exp(-(a * x + b)))

    x_data = np.array(x_data) + offset
    # Fit the sigmoid function to the scatter plot data
    popt, pcov = curve_fit(sigmoid, x_data, y_data, maxfev=100000, method = 'trf')
    
    # Generate x values for the fitted curve
    x_fit = np.linspace(0, max(x_data), 40)
    # Use the optimized parameters to generate y values for the fitted curve
    y_fit = sigmoid(x_fit, *popt)    
    # first find the target x in the given scatter plots
    if max(y_data) < target_y:
        
        # Use the optimized parameters to generate y values for the fitted curve
        y_fit = sigmoid(x_fit, *popt)
        
        while y_fit[-1] < target_y:
            x_fit = np.append(x_fit, x_fit[-1] + 1)
            y_fit = np.append(y_fit, sigmoid(x_fit[-1], *popt))
            
            # Break the loop if the condition is met
            if y_fit[-1] >= target_y:
                break
    
    # Generate x values for the fitted curve
    # Find the index of the target y-value
    target_y_index = np.argmin(np.abs(y_fit - target_y))
        
    # Retrieve the corresponding x-value
    target_x = x_fit[target_y_index]
    
    if not by_freq:     # assign the colors if plotting in one single fig
        
        plt.scatter(x_data, y_data, c= color)
        # plot until it has reached the target x
        plt.plot(x_fit, y_fit, linewidth=3.5,  linestyle = style, color = color,
                  label= label + f': {target_x:.2f}')
        
    else:     # if not, decided by legend labels
        
        plt.scatter(x_data, y_data)
        # plot until it has reached the target x
        plt.plot(x_fit, y_fit, linewidth=3.5,  linestyle = style,
                  label= label + f': {target_x:.2f}')
        
    plt.ylim(0, 1)
    # Marking the point where y reaches the target value
    plt.axvline(x=int(target_x), linestyle='dotted')
    header_lst = ['Label',"Month","Slope" , "Weighted_offset" ]
    # return the optimized parameters of the sigmoid function
    para_frame = pd.DataFrame([label,target_x,popt[0],popt[1]]).T
    para_frame.columns = header_lst
    plt.xlabel('(Pseudo) age in month', fontsize=15)
    plt.ylabel('Proportion of children/models', fontsize=15)
    plt.tick_params(axis='both', labelsize=10)
    return para_frame
    

   
def fit_log(x_data, y_data, label,color):
    '''
    fit sigmoid curve of extrapolated exp vocab

    '''

    def log_curve(x, a, b):
        return a * np.log2(x) + b

    # Fit the sigmoid function to the scatter plot data
    popt, pcov = curve_fit(log_curve, x_data, y_data, maxfev=100000, method = 'trf')
    
    # Generate x values for the fitted curve
    x_fit = np.linspace(0, max(x_data), 40)
    # Use the optimized parameters to generate y values for the fitted curve
    y_fit = log_curve(x_fit, *popt)    
    # first find the target x in the given scatter plots
    plt.scatter(x_data, y_data, c= color)
    # plot until it has reached the target x
    plt.plot(x_fit, y_fit, linewidth=3.5, color = color,label= label)
    plt.xlabel('Median freq', fontsize=15)
    plt.ylabel('Estimated months', fontsize=15)
    plt.tick_params(axis='both', labelsize=10)
    
    
    
  
def plot_exp(model_dir, target_frame, exp_threshold, label
             ,extrapolation,target_y,color_dict,curve_label,by_freq = False):
    

    def get_score(target_frame,seq_frame,threshold):
    
    
        '''
        get the weighted score based on different frequency range
        '''
        
        overlapping_words = [col for col in target_frame['word'].tolist() if col in seq_frame.index.tolist()]
        # get the subdataframe
        selected_frame = seq_frame.loc[overlapping_words]
        
        # use weighted class to decrase the effect of the unbalanced dataset
        extra_word_lst = [element for element in target_frame['word'].tolist() if element not in overlapping_words]
        for word in extra_word_lst:
            selected_frame.loc[word] = [0] * selected_frame.shape[1]
         
        # get the score based on theshold
        score_frame_all = selected_frame.applymap(lambda x: 1 if x >= threshold else 0)
        avg_values = score_frame_all.mean()
        
        return score_frame_all, avg_values
    
    
    def load_exp(seq_frame_unprompted,target_frame,exp_threshold):
        
        seq_frame_unprompted = seq_frame_unprompted.rename_axis('Index')
        avg_values_unprompted_lst = []
        score_frame_unprompted = pd.DataFrame()
        # decompose the results by freq groups
        for freq in set(list(target_frame['group'].tolist())):
            word_group = target_frame[target_frame['group']==freq]
            # take the weighted average by the proportion of different frequency types
            score_frame_unprompted_temp, avg_values_unprompted = get_score(word_group,seq_frame_unprompted,exp_threshold)
            score_frame_unprompted = pd.concat([score_frame_unprompted,score_frame_unprompted_temp])
            avg_values_unprompted_lst.append(avg_values_unprompted.values)
            
        arrays_matrix = np.array(avg_values_unprompted_lst)

        # Calculate the average array along axis 0
        avg_unprompted = np.mean(arrays_matrix, axis=0)

        return score_frame_unprompted, avg_unprompted
    
    # read the score frames
    seq_frame_all = pd.read_csv(model_dir + label + '.csv', index_col=0)
    score_frame_unprompted, avg_unprompted = load_exp(
            seq_frame_all, target_frame, exp_threshold)
    
    month_list_unprompted = [int(x) for x in score_frame_unprompted.columns]
    
    if not extrapolation:
        
        sns.lineplot(month_list_unprompted, avg_unprompted,
                      color=color_dict[label], linewidth=3, label=curve_label) 
        para_dict = None

    else:
        print('Plotting extrapolated figures of ' + label)
        
        para_dict = fit_sigmoid(month_list_unprompted, avg_unprompted,
                                    target_y,0,curve_label,color_dict[label],by_freq=by_freq)
        
        print('Have finished extrapolation of ' + label)
        
       
    
    # set the limits of the x-axis for each line
    if not extrapolation:
        plt.xlim(0, 36)
        plt.ylim(0, 1)
    
    
    return para_dict

