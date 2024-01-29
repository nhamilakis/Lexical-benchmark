#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 22:36:20 2023

@author: jliu
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit


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
    
    score_frame_all.to_csv('score_machine_exp.csv')
    return score_frame_all, avg_values



def get_score_CHILDES(freq_frame,threshold):
    
    '''
    get scores of the target words in Wordbank 
    input: word counts in each chunk/month and the stat of the number of true words as well as the true data proportion
    output: a dataframe with each word count  

    we have the weighed score in case the influence of different proportions of frequency bands      
    '''
     
    freq_frame = freq_frame.drop(columns=['word', 'group_original'])
    
    # get the score based on theshold
    score_frame = freq_frame.applymap(lambda x: 1 if x >= threshold else 0)
    
    avg_values = score_frame.mean()
    return score_frame,avg_values


def load_CDI(human_result):
    
    '''
    load human CDI data for figure plot
    '''
    
    size_lst = []
    month_lst = []
    
    n = 0
    while n < human_result.shape[0]:
        size_lst.append(human_result.iloc[n])
        headers_list = human_result.columns.tolist()
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
    return data_frame_final



def load_accum(accum_all,accum_threshold):
    
    df = accum_all[accum_all['threshold']==accum_threshold]
    accum_result = df.groupby(['month'])['Lexical score'].mean().reset_index()
    
    return accum_result




def load_exp(seq_frame_unprompted,target_frame,by_freq,exp_threshold):
    
    
    seq_frame_unprompted = seq_frame_unprompted.rename_axis('Index')
    
    if not by_freq:
        avg_values_unprompted_lst = []
        # decompose the results by freq groups
        for freq in set(list(target_frame['group'].tolist())):
            word_group = target_frame[target_frame['group']==freq]
            # take the weighted average by the proportion of different frequency types
            score_frame_unprompted, avg_values_unprompted = get_score(word_group,seq_frame_unprompted,exp_threshold)
            
            avg_values_unprompted_lst.append(avg_values_unprompted.values)
        
        arrays_matrix = np.array(avg_values_unprompted_lst)

        # Calculate the average array along axis 0
        avg_unprompted = np.mean(arrays_matrix, axis=0)

        
    # or we just read single subdataframe
    else:
        score_frame_unprompted, avg_unprompted = get_score(target_frame,seq_frame_unprompted,exp_threshold)
    
    return score_frame_unprompted, avg_unprompted 


def fit_curve_backup(x_data,y_data_temp,target_y,color,input_type):
    
    '''
    fit the extrapolation curves for the selected points
    '''
    
    # remove zeros to avoid log errors

    y_data = [0.01 if x == 0.0 else x for x in y_data_temp]     
    
    # Apply logarithmic transformation to the x and y data
    log_x = np.log(x_data)
    log_y = y_data
    
    
    # Fit a linear regression line to the log-transformed data
    slope, intercept, _, _, _ = linregress(log_x, log_y)
    
    # Create points for the fitted line on the log scale
    log_x_fit = np.linspace(min(log_x), max(log_x), 100)
    log_y_fit = slope * log_x_fit + intercept
    
    
    # Extrapolate the x-values until the target y-value is reached
    log_x_target = (np.log(target_y) - intercept) / slope
    x_target = np.exp(log_x_target)
    
    # Transform back from logarithmic scale to original scale
    x_fit = np.exp(log_x_fit)
    
    # Plotting the original data, the fitted line, and the extrapolated part
    plt.scatter(x_data, y_data, color=color)
    plt.plot(x_fit, np.exp(log_y_fit), color=color)     # only extend this line for extrapolation
    
    # this step is weird; why not just connect all the dots and add the extrapolations? 
    plt.plot([x_data[-1], x_target], [y_data[-1], target_y], linestyle='dashed', color=color)
    plt.axvline(x=x_target, color=color, linestyle='dotted', label= input_type + f': x = {x_target:.2f}')
    
    

def fit_log(x_data, y_data, target_y,color,label):
    
    
    # Define the logarithmic function
    '''
    def logarithmic(x, a, b):
        return a * np.log(x) + b
    '''
    def logarithmic(x, a):
           return a * np.log2(x)
    # Fit the logarithmic function to the scatter plot data
    popt, pcov = curve_fit(logarithmic, x_data, y_data)
    
    # Generate x values for the fitted curve
    x_fit = np.linspace(min(x_data), max(x_data), 1000)
    
    # Use the optimized parameters to generate y values for the fitted curve
    y_fit = logarithmic(x_fit, *popt)

    while y_fit[-1] < target_y:
        x_fit = np.append(x_fit, x_fit[-1] + 100)
        y_fit = np.append(y_fit, logarithmic(x_fit[-1], *popt))
        
        # Break the loop if the condition is met
        if y_fit[-1] >= target_y:
            break

    # Find the index of the target y-value
    target_y_index = np.argmin(np.abs(y_fit - target_y))
    
    # Retrieve the corresponding x-value
    target_x = x_fit[target_y_index]
    
    plt.scatter(x_data, y_data, c=color)
    plt.plot(x_fit, y_fit, linewidth=3.5, color=color, label= label + f': month = {target_x:.2f}')
    # Marking the point where y reaches the target value
    plt.axvline(x=int(target_x), color=color, linestyle='dotted')

    # return the optimized parameters of the sigmoid function
    para_dict = {}
    
    return para_dict

    

def fit_sigmoid(x_data, y_data, target_y, offset,color,label):
    '''
    fit sigmoid curve of extrapolated exp vocab

    '''

    def sigmoid(x, A, B, C):
        return 1 / (1 + np.exp(-(x - A) / B)) + C

    x_data = np.array(x_data) + offset
    # Fit the sigmoid function to the scatter plot data
    popt, pcov = curve_fit(sigmoid, x_data, y_data, maxfev=10000)

    # Generate x values for the fitted curve
    x_fit = np.linspace(min(x_data), max(x_data), 500)

    # Use the optimized parameters to generate y values for the fitted curve
    y_fit = sigmoid(x_fit, *popt)

    while y_fit[-1] < target_y:
        x_fit = np.append(x_fit, x_fit[-1] + 10000)
        y_fit = np.append(y_fit, sigmoid(x_fit[-1], *popt))
        
        # Break the loop if the condition is met
        if y_fit[-1] >= target_y:
            break

    # Find the index of the target y-value
    target_y_index = np.argmin(np.abs(y_fit - target_y))
    
    # Retrieve the corresponding x-value
    target_x = x_fit[target_y_index]
    
    plt.scatter(x_data, y_data, c=color)
    plt.plot(x_fit, y_fit, linewidth=3.5, color=color, label= label + f': month = {target_x:.2f}')
    # Marking the point where y reaches the target value
    plt.axvline(x=int(target_x), color=color, linestyle='dotted')

    # return the optimized parameters of the sigmoid function
    para_dict = {'Type':label,"Center": popt[0], "Width": popt[1], "Plateau": popt[2]}
    
    return para_dict
    



