#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: jliu

'''
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from plot_final_util import load_CDI,fit_sigmoid,plot_exp,fit_log,plot_exp_freq
import sys
import argparse



def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Plot all figures in the paper')

    parser.add_argument('--lang_lst', default=['AE'],
                        help='languages to test: AE, BE or FR')

    parser.add_argument('--vocab_type_lst', default=['exp'],
                        help='which type of words to evaluate; recep or exp')

    parser.add_argument('--model_dir', type=str, default='/data/Machine_CDI/Lexical-benchmark_output/Final_scores/Model_eval/exp/count/',
                        help='directory of results to plot')

    parser.add_argument('--human_dir', type=str, default="/data/Machine_CDI/Lexical-benchmark_output/test_set/matched_set/char/bin_range_aligned/6_audiobook_aligned/",
                        help='directory of CDI references')
    
    parser.add_argument('--fig_dir', type=str, default="Final_scores/Figures/",
                        help='directory of human CDI')

    parser.add_argument('--exp_threshold', type=int, default=60,
                        help='threshold for expressive vocab')

    parser.add_argument('--accum_threshold', type=int, default=60,
                        help='threshod for model and adult inputs')
    
    parser.add_argument('--color_dict', type=int, default={'speech': 'Blue','phones':'Green',
                        'human':'Red','child':'Orange','unprompted_0.3':'Grey','unprompted_0.6':'Blue',
                        'unprompted_1.0':'Green','unprompted_1.5':'Purple','Adult':'Orange'},
                        help='thresho, offsetld for accumulator model')
    
    parser.add_argument('--age_range', default=[6,96],
                        help='Age range to be included')
    
    parser.add_argument('--by_freq', default=True,
                        help='whether to decompose the results by frequency bands')

    parser.add_argument('--extrapolation', default=False,
                        help='whether to plot the extrapolation figure')
    
    parser.add_argument('--freq_analysis', default=False,
                        help='whether to analyze freq sensitivity')
    
    parser.add_argument('--target_y', type=float, default=0.8,
                        help='target y to reach for extrapolation')
    
    parser.add_argument('--condition', type=str, default='test',
                        help='either to investigate test set or exposure set')
    
    parser.add_argument('--analysis_type', type=str, default='test',
                        help='linear, log_linear, log_log')
    
    return parser.parse_args(argv)

sns.set_style('whitegrid')


def test_thre(human_dir,model_dir,exp_threshold_range,lang,label,color_dict):
    
    human_frame = pd.read_csv(human_dir + 'human_' + lang + '_exp.csv')
    human_result, prop_lst = load_CDI(human_frame)
    # plot the curve results
    month_lst = [int(x) for x in human_result.columns]

    sns.lineplot(month_lst, prop_lst,
                 color='black', linewidth=3.5, label='human')

    # load speech-based and phones-based models
    if label.startswith('child') or label.startswith('adult'):
        target_frame = human_frame
    else:
        target_frame = pd.read_csv(human_dir + 'machine_' + lang + '_exp.csv')
    # group by different frequencies

    for exp_threshold in exp_threshold_range:

        plot_exp(model_dir, target_frame, exp_threshold, label,
                 False, 0, color_dict, 'threshold: ' + str(exp_threshold), True, x_range = [0,36])

    
    # concatenate the result
    plt.xlabel('(Pseudo) age in month', fontsize=15)
    plt.ylabel('Proportion of children/models', fontsize=15)

    plt.title('{} exp vocab'.format(lang), fontsize=15, fontweight='bold')

    fig_dir = 'Final_scores/Figures/stat/exp_threshold/'
    # save the dataframe

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig_path = fig_dir + label + '_' + lang + '_.png'
    plt.savefig(fig_path, dpi=800)
    plt.show()
    

def plot_all(vocab_type, human_dir,model_dir, fig_dir, accum_threshold, exp_threshold
              ,lang,extrapolation,color_dict,target_y,condition,set_lst,age_range):

    model_dir = model_dir + lang + '/'
    # plot human CDI
    human_frame = pd.read_csv(human_dir + 'human_' + lang + '_' + vocab_type + '.csv')
    human_result,prop_lst = load_CDI(human_frame)
        # plot the curve results
    month_lst = [int(x) for x in human_result.columns]
        
    if not extrapolation:
        sns.lineplot(month_lst,prop_lst,color=color_dict['human'], linewidth=3.5, label='human')
    else:
        # only shows the by_month prop here 
        fit_sigmoid(month_lst,prop_lst, target_y,True,'human',color_dict['human']
                                                    ,by_freq=False)
    # TODO: add recep vocab in the future
    if vocab_type == 'exp':
        
        target_frame = pd.read_csv(human_dir + 'machine_' + lang + '_' + vocab_type + '.csv' )
        # plot the results recursively
        for score in set_lst:
            # compare with different reference data
            if score.startswith('child') or score.startswith('adult'):  
                target = human_frame
            else:
                target = target_frame
                
            plot_exp(model_dir, target, exp_threshold, score,age_range
                      ,extrapolation,target_y,color_dict, score)
           
    # concatenate the result
    plt.xlabel('(Pseudo) age in month', fontsize=15)
    plt.ylabel('Proportion of children/models', fontsize=15)
        
    plt.title('{} {} vocab'.format(
        lang, vocab_type), fontsize=15, fontweight='bold') 
    
    if extrapolation:
        parent_dir = 'extrapolation'
        # convert into dataframe   
    else:    
        parent_dir = 'stat'
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('(Pseudo) age in month', fontsize=15)
    plt.ylabel('Proportion of children/models', fontsize=15)
    fig_dir = 'Final_scores/Figures/' + parent_dir + '/avg/' 
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig_path = fig_dir + lang + '_' + vocab_type + '_' + condition+ '_' + str(exp_threshold) +'.png'
    plt.savefig(fig_path, dpi=800)
    plt.show()

    
   


def plot_by_freq(vocab_type,human_dir,model_dir,fig_dir,accum_threshold,exp_threshold
                 ,lang,set_type,extrapolation,target_y,color_dict,age_range):
    
    '''
    plot different freq bands in one figure； with/out extrapolation
    '''
    
    sns.set_style('whitegrid')
    
    model_dir = model_dir + lang + '/'
    # plot human CDI
    human_frame_all = pd.read_csv(human_dir + 'human_' + lang + '_' + vocab_type + '.csv')
    freq_lst = set(human_frame_all['group'].tolist())
    target_frame = pd.read_csv(human_dir + 'machine_' + lang + '_' + vocab_type + '.csv' )
    para_frame = pd.DataFrame()
   
    for freq in freq_lst:
        
        if set_type == 'human':
            
            # read the results recursively
            label= 'human'
            # read the results recursively
            human_frame,frame = human_frame_all[human_frame_all['group']==freq]
            
            human_result,prop_lst = load_CDI(human_frame)
            # plot the curve results
            month_lst = [int(x) for x in human_result.columns]
            # map the legend label to freq median
            median_freq = "{:.2f}".format(human_frame['freq_median'].tolist()[0])
            
            if not extrapolation:
                sns.lineplot(month_lst,prop_lst, linewidth=3.5, label=freq)
            else:
                # only shows the by_month prop here 
                fit_sigmoid(month_lst,prop_lst, target_y,0,str(freq),color_dict[label]
                                                        ,by_freq=True)
        
        if set_type.startswith('child') or set_type.startswith('adult'):  
            target = human_frame_all[human_frame_all['group']==freq]
        else:
            target = target_frame[target_frame['group']==freq]
        
        
        
        plot_exp(model_dir, target, exp_threshold, set_type,age_range
                  ,extrapolation,target_y,color_dict,str(freq))
        
        '''
        if extrapolation:     # record the ex
            para['group'] = freq
            para_frame = pd.concat([para_frame,para])
        '''
    plt.title('{} {} vocab ({})'.format(
         lang, vocab_type,set_type), fontsize=15, fontweight='bold')
     
    
    if not os.path.exists(fig_dir):
         os.makedirs(fig_dir)
    fig_path = fig_dir + lang + '_' + vocab_type + '_' + set_type+ '_' + str(exp_threshold) +'.png'
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    plt.savefig(fig_path, dpi=800)
    plt.show()          
    return para_frame
        
      
    
def freq_sensitivity(lang, vocab_type,color_dict,condition,exp_threshold):
    
    '''
    input: a dataframe with freq bins adn estimated months
    output: curves in months
    '''
    para_all = pd.read_csv('Final_scores/Figures/extrapolation/by_freq/exp_' + condition + str(exp_threshold) + '.csv')
    
    para_frame = para_all[para_all['lang']==lang]
    para_frame_grouped = para_frame.groupby(['set'])
    for group, para_frame_group in para_frame_grouped:
        # plot the curve 
        # convert median back to linear 
        median_lst = [10 ** x for x in para_frame_group['Label']]
        #sns.lineplot(median_lst,para_frame_group['Month'],linewidth=3.5, label=group)
        #plt.scatter(median_lst,para_frame_group['Month'], label=group)
        fit_log(median_lst,para_frame_group['Month'],group,color_dict[group])

    plt.title('{} {} vocab'.format(
        lang, vocab_type), fontsize=15, fontweight='bold') 
    legend_loc = 'upper right'
    plt.legend(loc=legend_loc)
    plt.ylim(-10, 90)
    fig_dir = 'Final_scores/Figures/extrapolation/by_freq/' 
    if not os.path.exists(fig_dir):
         os.makedirs(fig_dir)
    fig_path = fig_dir + lang + '_' + vocab_type + '_' + condition + '_sensitivity_' + str(exp_threshold) + '.png'
    plt.savefig(fig_path, dpi=800)
    plt.show()
    plt.clf()
    

def main(argv):
    # Args parser
    args = parseArgs(argv)
    accum_threshold = args.accum_threshold
    exp_threshold = args.exp_threshold
    color_dict = args.color_dict
    condition = args.condition
    human_dir = args.human_dir
    model_dir = args.model_dir
    target_y = args.target_y
    extrapolation = args.extrapolation
    age_range = args.age_range
    fig_dir = args.fig_dir
   
    
    if condition == 'test':
        set_lst = ['child','unprompted_0.3','unprompted_0.6','unprompted_1.0','unprompted_1.5']
                    
    elif condition == 'exposure':
        set_lst = ['train','Adult']
    
    
    for vocab_type in args.vocab_type_lst:
        for lang in args.lang_lst:
             
            if not args.by_freq:
                plot_all(vocab_type, human_dir,model_dir, fig_dir, accum_threshold, exp_threshold
                              ,lang,extrapolation,color_dict,target_y,condition,set_lst,age_range)
            if args.by_freq:
                
                para_frame = pd.DataFrame()
                for set_type in set_lst:
                    
                    para_set = plot_by_freq(vocab_type,human_dir,model_dir,fig_dir,accum_threshold,exp_threshold
                                 ,lang,set_type,args.extrapolation,args.target_y,color_dict,age_range)
                    
                                
                '''
                if args.by_freq:
                    
                    if args.extrapolation:
                        para_frame = pd.DataFrame()
                        for set_type in set_lst:
                            
                            para_set = plot_by_freq(vocab_type,human_dir,model_dir,test_set,accum_threshold,exp_threshold
                                         ,lang,set_type,args.extrapolation,args.target_y,color_dict)
                            
                            para_set['set'] = set_type
                            
                            para_frame = pd.concat([para_frame,para_set])
                        para_frame['lang'] = lang
                        para_all = pd.concat([para_all,para_frame])
                        para_all.to_csv('Final_scores/Figures/extrapolation/by_freq/exp_' + condition + str(args.exp_threshold) +'.csv' )
                        
                    else:
                        for set_type in set_lst:
                            para_set = plot_by_freq(vocab_type,human_dir,model_dir,test_set,accum_threshold,exp_threshold
                                         ,lang,set_type,args.extrapolation,args.target_y,color_dict)
                elif args.freq_analysis:
                    freq_sensitivity(lang, vocab_type,color_dict,condition,exp_threshold)
                    
                else:
                    
                    plot_all(vocab_type, human_dir,model_dir, test_set, accum_threshold
                         ,exp_threshold,lang,args.extrapolation,color_dict,args.target_y,condition) 
                    
                    para_frame = plot_all(vocab_type, human_dir,model_dir, test_set, accum_threshold
                         ,exp_threshold,lang,args.extrapolation,color_dict,args.target_y,condition) 
                    
                    freq = 'avg'
                    para_frame['lang'] = lang
                    para_frame['freq'] = freq
                    para_all = pd.concat([para_all,para_frame])
                    '''
                
                
                
                        
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)