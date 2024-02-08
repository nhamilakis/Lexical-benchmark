#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 22:36:20 2023

@author: jliu
"""

import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
#from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.optimize import fsolve

target_frame = pd.read_csv('/data/Machine_CDI/Lexical-benchmark_output/test_set/matched_set/char/bin_range_aligned/backup/6_audiobook_aligned/' + 'machine_BE_exp.csv')
seq_frame = pd.read_csv('/data/Machine_CDI/Lexical-benchmark_output/Final_scores/Model_eval/BE/exp/unprompted.csv')



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


def load_CDI(human_frame):
    
    '''
    load human CDI data for figure plot; x_axis; y_axis data 
    '''
    start_idx = human_frame.columns.get_loc('category')
    end_idx = human_frame.columns.get_loc('word')
    human_result = human_frame.iloc[:, start_idx+1:end_idx]
    
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
    
    return data_frame_final, data_frame_final["month"], data_frame_final["Proportion of acquired words"]



def load_exp(seq_frame_unprompted,target_frame,exp_threshold):
    
    
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
    month_list_unprompted = [int(x) for x in score_frame_unprompted.columns]
    
    return score_frame_unprompted, month_list_unprompted, avg_unprompted
    
  

def load_accum(accum_all,accum_threshold):
    
    df = accum_all[accum_all['threshold']==accum_threshold]
    accum_result = df.groupby(['month'])['Lexical score'].mean().reset_index()
    
    return accum_result




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


'''
exp_threshold = 60
model_dir = "Final_scores/Model_eval/AE/" + 'exp/matched/'
freq_lst = ['high', 'low']
sub_dict = {0: 'low', 1: 'low',2: 'low', 3: 'high',4: 'high', 5: 'high'}

# read the generated file
target_dir = "Final_scores/Human_eval/CDI/" + 'AE/exp/'
for file in os.listdir(target_dir):
    target_frame = pd.read_csv(target_dir + '/' + file)
target_frame['group']=target_frame['group'].replace(sub_dict)

for freq in freq_lst:      
    # unprompted generation
       
    seq_frame_all = pd.read_csv(model_dir + 'unprompted.csv', index_col=0)
    # get the sub-dataframe by frequency  
    score_frame_all, avg_unprompted_all = load_exp(seq_frame_all,target_frame,True,exp_threshold)   
        
    word_group = target_frame[target_frame['group']==freq]['word'].tolist()
    score_frame = score_frame_all.loc[word_group]
    avg_values = score_frame.mean()
    month_list_unprompted = [int(x) for x in score_frame.columns]

x_data = month_list_unprompted


score_frame = pd.read_csv('AE_generation.csv')
score_frame = score_frame.drop(score_frame.columns[0], axis=1)
y_data = score_frame.mean()
x_data = [int(x) for x in score_frame.columns]

offset = 5
color = 'grey'
style = 'solid'
target_y = 0.8
'''


# def fit_sigmoid(x_data, y_data, target_y, offset,color,label,style):
#     '''
#     fit sigmoid curve of extrapolated exp vocab

#     '''

#     def sigmoid(x, a, b):
#         return 1 / (1 + np.exp(-(a * x + b)))

#     x_data = np.array(x_data) + offset
#     # Fit the sigmoid function to the scatter plot data
#     popt, pcov = curve_fit(sigmoid, x_data, y_data, maxfev=100000, method = 'dogbox')
    
#     # Given the target y value, set up an equation to solve for x
#     def equation_to_solve(x, target_y, *popt):
#         return sigmoid(x, *popt) - target_y
    
#     # Use fsolve to solve the equation for x
#     def find_x_for_target_y(target_y, popt):
#         x_initial_guess = 0  # Initial guess for x
#         x_solution = fsolve(equation_to_solve, x_initial_guess, args=(target_y, *popt))
#         return x_solution[0]
    
#     # first find the target x in the given scatter plots
#     if max(y_data) > target_y:
#         # Generate x values for the fitted curve
#         x_fit = np.linspace(min(x_data), max(x_data), 10000)
        
#         # Use the optimized parameters to generate y values for the fitted curve
#         y_fit = sigmoid(x_fit, *popt)
        
#         # Find the index of the target y-value
#         target_y_index = np.argmin(np.abs(y_fit - target_y))
        
#         # Retrieve the corresponding x-value
#         target_x = x_fit[target_y_index]
#     else:
        
#         target_x = find_x_for_target_y(target_y, popt)
        
#         # Generate x values for the fitted curve
#         x_fit = np.linspace(min(x_data), target_x, 500)
    
#         # Use the optimized parameters to generate y values for the fitted curve
#         y_fit = sigmoid(x_fit, *popt)
    
    
#     # Generate x values for the fitted curve
#     x_fit = np.linspace(max(x_data), 500, 500)

#     # Use the optimized parameters to generate y values for the fitted curve
#     y_fit = sigmoid(x_fit, *popt)
        
#     plt.scatter(x_data, y_data, c=color)
    
    
    
#     plt.plot(x_fit, y_fit, linewidth=3.5, color=color, linestyle = style,
#               label= label + f': month = {target_x:.2f}')
#     '''
    
#     plt.plot(x_fit, y_fit, linewidth=3.5, color=color, linestyle = style,
#               label= label)
#     '''
#     # Marking the point where y reaches the target value
#     #plt.axvline(x=int(target_x), color=color, linestyle='dotted')

#     # return the optimized parameters of the sigmoid function
#     para_dict = {'Type':label,"Center": popt[0], "Width": popt[1]}
    
    
#     return para_dict, 0 
    



def fit_sigmoid(x_data, y_data, target_y, offset,label,style):
    '''
    fit sigmoid curve of extrapolated exp vocab

    '''

    def sigmoid(x, a, b):
        return 1 / (1 + np.exp(-(a * x + b)))

    x_data = np.array(x_data) + offset
    # Fit the sigmoid function to the scatter plot data
    popt, pcov = curve_fit(sigmoid, x_data, y_data, maxfev=100000, method = 'trf')
    
    
    # Generate x values for the fitted curve
    x_fit = np.linspace(0, max(x_data), 50)
    # Use the optimized parameters to generate y values for the fitted curve
    y_fit = sigmoid(x_fit, *popt)    
    # first find the target x in the given scatter plots
    if max(y_data) < target_y:
        
        # Use the optimized parameters to generate y values for the fitted curve
        y_fit = sigmoid(x_fit, *popt)
        
        while y_fit[-1] < target_y:
            x_fit = np.append(x_fit, x_fit[-1] + 2)
            y_fit = np.append(y_fit, sigmoid(x_fit[-1], *popt))
            
            # Break the loop if the condition is met
            if y_fit[-1] >= target_y:
                break
    
    # Generate x values for the fitted curve
    # Find the index of the target y-value
    target_y_index = np.argmin(np.abs(y_fit - target_y))
        
    # Retrieve the corresponding x-value
    target_x = x_fit[target_y_index]
    plt.scatter(x_data, y_data)
    
    # plot until it has reached the target x
    plt.plot(x_fit, y_fit, linewidth=3.5,  linestyle = style,
              label= label + f': month = {target_x:.2f}')
    plt.ylim(0, 1)
    # Marking the point where y reaches the target value
    plt.axvline(x=int(target_x), linestyle='dotted')

    # return the optimized parameters of the sigmoid function
    para_dict = {'Type':label,"Center": popt[0], "Width": popt[1]}
    
    return para_dict, 0 
    


def get_linear(x_data, y_data, target_y,label):
    '''
    fit sigmoid curve of extrapolated exp vocab

    '''

    def linear(x, m, c):
        return m * x + c

    x_data = np.array(x_data)
    # Fit the sigmoid function to the scatter plot data
    popt, _ = curve_fit(linear, x_data, y_data, maxfev=10000)

    
    # Given the target y value, set up an equation to solve for x
    def equation_to_solve(x, target_y, *popt):
        return linear(x, *popt) - target_y
    
    # Use fsolve to solve the equation for x
    def find_x_for_target_y(target_y, popt):
        x_initial_guess = 0  # Initial guess for x
        x_solution = fsolve(equation_to_solve, x_initial_guess, args=(target_y, *popt))
        return x_solution[0]
    
    # first find the target x in the given scatter plots
    if max(y_data) > target_y:
        # Generate x values for the fitted curve
        x_fit = np.linspace(min(x_data), max(x_data), 1000)
        
        # Use the optimized parameters to generate y values for the fitted curve
        y_fit = linear(x_fit, *popt)
        
        # Find the index of the target y-value
        target_y_index = np.argmin(np.abs(y_fit - target_y))
        
        # Retrieve the corresponding x-value
        target_x = x_fit[target_y_index]
    else:
        
        target_x = find_x_for_target_y(target_y, popt)
        
        # Generate x values for the fitted curve
        x_fit = np.linspace(min(x_data), target_x, 500)
    
        # Use the optimized parameters to generate y values for the fitted curve
        y_fit = linear(x_fit, *popt)

    # return the optimized parameters of the sigmoid function
    para_dict = {'Type':label,'target_y':target_y,'estimated_month':target_x,
                 "Center": popt[0], "Width": popt[1], "Plateau":' NA'}
    
    
    return para_dict,x_fit,y_fit
    

