#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: jliu

color setting
human: red
freq-based:purpl -> pink for matched
model:
    1.speech/prompted (the most human like): speech
    2.phone: blue
    2.phoneme/unprompted: purple
by_freq: similar colors but different shape
    high: line
    low: dotted

'''
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from plot_final_util import load_CDI,fit_sigmoid,plot_exp,fit_log
import sys
import argparse

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Plot all figures in the paper')

    parser.add_argument('--lang_lst', default=['AE','BE'],
                        help='languages to test: AE, BE or FR')

    parser.add_argument('--vocab_type_lst', default=['exp'],
                        help='which type of words to evaluate; recep or exp')

    parser.add_argument('--testset_lst', default=['matched'],
                        help='which testset to evaluate')

    parser.add_argument('--model_dir', type=str, default='Final_scores/Model_eval/',
                        help='directory of machine CDI results')

    parser.add_argument('--human_dir', type=str, default="Final_scores/Human_eval/CDI/",
                        help='directory of human CDI')

    parser.add_argument('--fig_dir', type=str, default="Final_scores/Figures/",
                        help='directory of human CDI')

    parser.add_argument('--exp_threshold', type=int, default=60,
                        help='threshold for expressive vocab')

    parser.add_argument('--accum_threshold', type=int, default=200,
                        help='thresho, offsetld for accumulator model')
    
    parser.add_argument('--color_dict', type=int, default={'speech': 'Blue','phones':'Green',
                        'human':'Red','CHILDES':'Orange','unprompted':'Grey','prompted':'Blue'},
                        help='thresho, offsetld for accumulator model')

    parser.add_argument('--by_freq', default=True,
                        help='whether to decompose the results by frequency bands')

    parser.add_argument('--extrapolation', default=True,
                        help='whether to plot the extrapolation figure')
    
    parser.add_argument('--aggregate_freq', default=False,
                        help='whether to aggregate frequency bands into 2')
    
    parser.add_argument('--freq_analysis', default=False,
                        help='whether to analyze freq sensitivity')
    
    parser.add_argument('--target_y', type=float, default=0.8,
                        help='target y to reach for extrapolation')
    
    return parser.parse_args(argv)

sns.set_style('whitegrid')



def plot_all(vocab_type, human_dir,model_dir, test_set, accum_threshold, exp_threshold
              ,lang,extrapolation,color_dict,target_y):

    # plot the curve averaged by freq bands
    # plot human
    linestyle_lst = ['solid', 'dotted']
    
    n = 0
    for file in os.listdir(human_dir):
        label = 'human'
        
        human_frame = pd.read_csv(human_dir + '/' + file)
        human_result,prop_lst = load_CDI(human_frame)
        # plot the curve results
        month_lst = [int(x) for x in human_result.columns]
        
        if not extrapolation:
            sns.lineplot(month_lst,prop_lst,color=color_dict[label], 
                              linestyle=linestyle_lst[n], linewidth=3.5, label=label)
        else:
            # only shows the by_month prop here 
            para_human = fit_sigmoid(month_lst,prop_lst, target_y,0,label,color_dict[label]
                                                    ,by_freq=False,style=linestyle_lst[n])
            
        n += 1
    
    # load speech-based and phones-based models
    if vocab_type == 'recep':

        # plot accumulator model
        accum_all = pd.read_csv(model_dir + 'accum.csv')
        accum_result = load_accum(accum_all, accum_threshold)
        sns.lineplot(x="month", y="Lexical score", data=accum_result,
                          color="Green", linewidth=3, label='Accumulator')

        speech_result = pd.read_csv(model_dir + 'speech.csv')
        
        # remove the chunk in which there is only one word in each freq band
        # merge among similar freq band of the same chunk
        speech_result = speech_result.groupby(['month','chunk'])['mean_score'].mean().reset_index()
        sns.lineplot(x="month", y="mean_score", data=speech_result,       
                          color='Blue', linewidth=3.5, label='speech')     # show the freq band's level range 
        
        
        # plot speech-based model
        phones_result = pd.read_csv(model_dir + 'phones.csv')
        phones_result = phones_result.groupby(['month','chunk'])['mean_score'].mean().reset_index()
        sns.lineplot(x="month", y="mean_score", data=phones_result,
                          color="Purple", linewidth=3.5, label='phones')

    elif vocab_type == 'exp':
        
        # group by different frequencies
        
        para_CHILDES = plot_exp(model_dir, human_frame, exp_threshold, 'CHILDES'
                      ,extrapolation,target_y,color_dict, 'CHILDES')
        
        
        target_dir = human_dir.replace('CDI', test_set)
        target_frame = pd.read_csv(target_dir + '/median_aligned.csv' )
        
        para_unprompted = plot_exp(model_dir, target_frame, exp_threshold, 'unprompted'
                      ,extrapolation,target_y,color_dict, 'unprompted')
        
        para_prompted = plot_exp(model_dir, target_frame, exp_threshold, 'prompted'
                      ,extrapolation,target_y,color_dict, 'prompted')
    # concatenate the result
    if extrapolation:
        para_all = pd.concat([para_human,para_CHILDES, para_unprompted,para_prompted], ignore_index=True)
    plt.title('{} {} vocab'.format(
        lang, vocab_type), fontsize=15, fontweight='bold') 
    if args.extrapolation:
        parent_dir = 'extrapolation'
        # convert into dataframe   
    else:    
        parent_dir = 'stat'
        
    legend_loc = 'upper left'
    plt.legend(loc=legend_loc)
    fig_dir = 'Final_scores/Figures/' + parent_dir + '/' + freq + '/' 
    # save the dataframe
    para_all.to_csv(fig_dir + vocab_type + '_' + test_set+'.csv')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig_path = fig_dir + lang + '_' + vocab_type + '_' + test_set+'.png'
    plt.savefig(fig_path, dpi=800)
    
    plt.show()

    return para_all
    
    


def plot_by_freq(vocab_type,human_dir,model_dir,test_set,accum_threshold,exp_threshold
                 ,lang,set_type,extrapolation,target_y,color_dict):
    
    '''
    plot different freq bands in one figure； with/out extrapolation
    '''
    
    sns.set_style('whitegrid')
    
    for file in os.listdir(human_dir):
        human_frame_all = pd.read_csv(human_dir + '/' + file)
        freq_lst = set(human_frame_all['group'].tolist())
        
    target_dir = human_dir.replace('CDI', test_set)     # Final_scores\Human_eval\matched\AE\exp
    target_frame = pd.read_csv(target_dir + '/median_aligned.csv' )
    
    para_frame = pd.DataFrame()
    for freq in freq_lst:
        
        if set_type == 'human':
            
            # read the results recursively
            label= 'human'
            # read the results recursively
            human_frame = human_frame_all[human_frame_all['group']==freq]
            
            human_result,prop_lst = load_CDI(human_frame)
            # plot the curve results
            month_lst = [int(x) for x in human_result.columns]
            # map the legend label to freq median
            median_freq = "{:.2f}".format(human_frame['freq_median'].tolist()[0])
            
            if not extrapolation:
                sns.lineplot(month_lst,prop_lst, linewidth=3.5, label=freq)
            else:
                # only shows the by_month prop here 
                para = fit_sigmoid(month_lst,prop_lst, target_y,0,str(median_freq),color_dict[label]
                                                        ,by_freq=True)
         
             
        # plot accumulator model
        if set_type == 'accum':
                # accumulator model
                    accum_all_freq = pd.read_csv(model_dir + test_set +'/accum.csv')
                    accum_all = accum_all_freq[accum_all_freq['group']==freq]
                    accum_result = load_accum(accum_all,accum_threshold)
                    ax = sns.lineplot(x="month", y="Lexical score", data=accum_result, color="Green", linewidth=3, label = freq)
                
        if set_type == 'speech':
                    speech_result_all = pd.read_csv(model_dir + 'speech.csv')  
                    # seelct speech model based on the freq band
                    speech_result = speech_result_all[speech_result_all['group']==freq]
                    ax = sns.lineplot(x="month", y="mean_score", data=speech_result, linewidth=3,label= freq)
                    
        if set_type == 'phoneme':   
                    # plot phone-LSTM model
                    phone_result_all = pd.read_csv(model_dir + test_set + '/phones.csv')
                    # seelct speech model based on the freq band
                    phone_result = phone_result_all[phone_result_all['group_median']==freq]
                    ax = sns.lineplot(x="month", y="group", data=phone_result,linewidth=3, label= freq)
                    
                
        if set_type == 'CHILDES':
            # add human-estimation here
            human_frame = human_frame_all[human_frame_all['group']==freq]
            # map the legend label to freq median
            median_freq = "{:.2f}".format(human_frame['freq_median'].tolist()[0])
            para = plot_exp(model_dir, human_frame, exp_threshold, 'CHILDES'
                          ,extrapolation,target_y,color_dict,str(median_freq),by_freq=True,)
                        
        
        if set_type == 'unprompted':
                
            word_group = target_frame[target_frame['group']==freq]
            # map the legend label to freq median
            median_freq = "{:.2f}".format(word_group['freq_median'].tolist()[0])
            para = plot_exp(model_dir, word_group, exp_threshold, 'unprompted'
                          ,extrapolation,target_y,color_dict,str(median_freq),by_freq=True)
            
        if set_type == 'prompted':
                
            word_group = target_frame[target_frame['group']==freq]
            # map the legend label to freq median
            median_freq = "{:.2f}".format(word_group['freq_median'].tolist()[0])
            para = plot_exp(model_dir, word_group, exp_threshold, 'prompted'
                          ,extrapolation,target_y,color_dict,str(median_freq),by_freq=True)
        
        para['group'] = freq
        para_frame = pd.concat([para_frame,para])
        
    plt.title('{} {} vocab ({})'.format(
         lang, vocab_type,set_type), fontsize=15, fontweight='bold')
     
    fig_dir = 'Final_scores/Figures/extrapolation/by_freq/' 
    if not os.path.exists(fig_dir):
         os.makedirs(fig_dir)
    fig_path = fig_dir + lang + '_' + vocab_type + '_' + set_type+'.png'
    
    legend_loc = 'upper left'
    plt.legend(loc=legend_loc) 
    plt.savefig(fig_path, dpi=800)
    plt.show()          
    return para_frame
        
      
    
    
    

     
def aggre_freq(vocab_type,human_dir,model_dir,test_set,accum_threshold,exp_threshold
               ,lang,extrapolation,target_y,color_dict):
    
    '''
    plot mdifferent freq bands in one figure
    aggregate into 2 bands: high or low      
     
    '''
    style = ['solid','dotted']
    sns.set_style('whitegrid')
    sub_dict = {0: 'low', 1: 'low',2: 'low', 3: 'high',4: 'high', 5: 'high'}
    freq_lst = ['high', 'low']
    
    
    for file in os.listdir(human_dir):
            human_frame_all = pd.read_csv(human_dir + '/' + file)
            human_frame_all['group'] = human_frame_all['group'].replace(sub_dict)
            
            n = 0
            for freq in freq_lst:
                label= 'human'
                # read the results recursively
                human_frame = human_frame_all[human_frame_all['group']==freq]
                human_result,prop_lst = load_CDI(human_frame)
                # plot the curve results
                month_lst = [int(x) for x in human_result.columns]
                
                if not extrapolation:
                    if n == 0:
                        sns.lineplot(month_lst,prop_lst,color=color_dict[label], 
                                      linestyle=style[n], linewidth=3.5, label=label)
                    elif n == 1:
                        sns.lineplot(month_lst,prop_lst,color=color_dict[label], 
                                      linestyle=style[n], linewidth=3.5)
                else:
                    # only shows the by_month prop here
                    para_dict = fit_sigmoid(month_lst,prop_lst, target_y,0,label,color_dict[label]
                                                            ,by_freq=False,style=style[n])
                
                n += 1
    
    
    
    # if vocab_type == 'recep':
            
    #     # replace freq group
    #     accum_all_freq = pd.read_csv(model_dir +'/accum.csv')
        
    #     speech_result_all = pd.read_csv(model_dir + 'speech.csv')  
    #     phone_result_all = pd.read_csv(model_dir + '/phones.csv')
        
    #     # accum_all_freq['group'] = accum_all_freq['group'].replace(sub_dict)
    #     speech_result_all['group'] = speech_result_all['group'].replace(sub_dict)
    #     phone_result_all['group'] = phone_result_all['group'].replace(sub_dict)
        
    #     #accum_all_freq = accum_all_freq['group'].replace(sub_dict) 
        
        
    #     n = 0
    #     for freq in freq_lst:   
                
                
    #             accum_all = accum_all_freq[accum_all_freq['group']==freq]
    #             accum_result = load_accum(accum_all,accum_threshold)
    #             if n == 0:
    #                 ax = sns.lineplot(x="month", y="Lexical score", data=accum_result, color="Green", 
    #                               linewidth=3, label = 'Accum',linestyle = style[n])
                    
    #             else:
    #                 ax = sns.lineplot(x="month", y="Lexical score", data=accum_result, color="Green", 
    #                               linewidth=3, linestyle = style[n])
                
    #             # seelct speech model based on the freq band
    #             speech_result = speech_result_all[speech_result_all['group']==freq]
                
    #             speech_result = speech_result.groupby(['month','chunk'])['mean_score'].mean().reset_index()
    #             if extrapolation:
    #                 fit_log(speech_result.index.tolist(), speech_result.tolist(),target_y, 'Blue', 'speech')
    #             else:    
                    
    #                 if n == 0:
    #                     ax = sns.lineplot(x="month", y="mean_score", data=speech_result, linewidth=3,label= 'speech'
    #                               ,linestyle = style[n], color = 'Blue')
    #                 else:
    #                     ax = sns.lineplot(x="month", y="mean_score", data=speech_result, linewidth=3
    #                               ,linestyle = style[n], color = 'Blue')
     
                  
    #             # plot phone-LSTM model
                
    #             # seelct speech model based on the freq band
                
    #             phone_result = phone_result_all[phone_result_all['group']==freq]
    #             phone_result = phone_result.groupby(['month','chunk'])['mean_score'].mean().reset_index()
                
    #             if n == 0:
    #                 ax = sns.lineplot(x="month", y="mean_score", data=phone_result,linewidth=3, 
    #                                   label= 'phones',linestyle = style[n],color = 'Purple')
                    
    #             if n == 1:
    #                 ax = sns.lineplot(x="month", y="mean_score", data=phone_result,linewidth=3, 
    #                                   linestyle = style[n],color = 'Purple')  
                  
    #             n += 1
       
            
       
    if vocab_type == 'exp':
            
            # add human-estimation here
            CHILDES_frame = pd.read_csv('Final_scores/Model_eval/' + lang + '/exp/CDI/CHILDES.csv')
            # read the generated file
            target_dir = human_dir.replace('CDI', test_set)
            if test_type == 'aligned':
                target_frame = pd.read_csv(target_dir + '/median_aligned.csv' )
                
            else:
                if test_type == 'unaligned':
                    target_frame = pd.read_csv(target_dir + '/median_unaligned.csv' )
                    
            target_frame['group']=target_frame['group'].replace(sub_dict)
            CHILDES_frame['group_original'] = CHILDES_frame['group_original'].replace(sub_dict)
            
            
            n = 0
            # unprompted generations
            for freq in freq_lst:   
                
                # CHILDES 
                label = 'CHILDES'
                CHILDES_freq = CHILDES_frame[CHILDES_frame['group_original']==freq]
                CHIDES_result, avg_values = get_score_CHILDES(CHILDES_freq, exp_threshold)
                month_list_CHILDES = [int(x) for x in CHIDES_result.columns]
                if extrapolation:
                    
                    para_dict = fit_sigmoid(month_list_CHILDES, avg_values, target_y,0,label
                                                            ,color_dict[label],by_freq=False,style=style[n])
                    
                else:    
                    
                    if n == 0:
                        ax = sns.lineplot(month_list_CHILDES, avg_values,linestyle = style[n],
                                      linewidth=3, label= 'CHILDES', color = 'Orange')
                    elif n== 1:
                        ax = sns.lineplot(month_list_CHILDES, avg_values,linestyle = style[n],
                                      linewidth=3,color = 'Orange')
                
                n += 1 
            
                
            
            n = 0
            # unprompted generations
            for freq in freq_lst:      
                 # unprompted generation
                 label = 'unprompted'
                 seq_frame_all = pd.read_csv(model_dir + 'unprompted.csv', index_col=0)
                 # get the sub-dataframe by frequency  
                 score_frame_all, avg_unprompted_all = load_exp(seq_frame_all,target_frame,False,exp_threshold)   
                    
                 word_group = target_frame[target_frame['group']==freq]['word'].tolist()
                 score_frame = score_frame_all.loc[word_group]
                 avg_values = score_frame.mean()
                 month_list_unprompted = [int(x) for x in score_frame.columns]
                
                 score_frame.to_csv(lang + '_generation.csv')
                 if extrapolation:
                     para_dict,corresponding_x = fit_sigmoid(month_list_unprompted, avg_values.values,
                                                 target_y,0,label,color_dict[label],by_freq=False,style = style[n])
                
                 else:   
                     if n == 0:
                        
                         ax = sns.lineplot(month_list_unprompted, avg_values.values, linewidth=3
                                   , label= 'unprompted', color = 'Grey',linestyle = style[n])
                        
                     elif n == 1:
                         ax = sns.lineplot(month_list_unprompted, avg_values.values, linewidth=3
                                   , color = 'Grey',linestyle = style[n])
                         
                         
                
                 n += 1
                
            n = 0  
            for freq in freq_lst:      
                 # unprompted generation
                 label = 'prompted'
                 seq_frame_all = pd.read_csv(model_dir + 'prompted.csv', index_col=0)
                 # get the sub-dataframe by frequency  
                 score_frame_all, avg_unprompted_all = load_exp(seq_frame_all,target_frame,False,exp_threshold)   
                    
                 word_group = target_frame[target_frame['group']==freq]['word'].tolist()
                 score_frame = score_frame_all.loc[word_group]
                 avg_values = score_frame.mean()
                 month_list_unprompted = [int(x) for x in score_frame.columns]
                
                 score_frame.to_csv(lang + '_generation.csv')
                 if extrapolation:
                     para_dict,corresponding_x = fit_sigmoid(month_list_unprompted, avg_values.values,
                                                 target_y,0,label,color_dict[label],by_freq=False,style = style[n])
                
                 else:   
                     if n == 0:
                         ax = sns.lineplot(month_list_unprompted, avg_values.values, linewidth=3
                                   , label= 'prompted', color = 'Blue',linestyle = style[n])
                        
                     elif n == 1:
                         ax = sns.lineplot(month_list_unprompted, avg_values.values, linewidth=3
                                   , color = 'Blue',linestyle = style[n])
                        
                
                 n += 1
       
    # set the limits of the x-axis for each line
    if not extrapolation:
        for line in ax.lines:
            plt.xlim(0,36)
            plt.ylim(0,1)
    
    plt.title('{} {} vocab'.format(lang,vocab_type), fontsize=15, fontweight='bold')
    
    ylabel = 'Proportion of children/models'
    xlabel = '(Pseudo) age in month'
    
    plt.ylabel(ylabel, fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.tick_params(axis='both')
    
    # set legend location to avoid shade other curves
    
    if vocab_type == 'exp':
        legend_loc = 'upper left'
        # Create proxy artists for the legend labels
        plt.legend(fontsize='small',loc=legend_loc)
        
    else:
        legend_loc = 'upper right'
        plt.legend(fontsize='small', loc=legend_loc, bbox_to_anchor=(1, 0.9))
    
    plt.savefig('Final_scores/Figures/freq/' + lang + '_' + vocab_type + '_freq.png',dpi=800)
    plt.show()
    
    
  
    
def freq_sensitivity(lang, vocab_type,color_dict):
    
    '''
    input: a dataframe with freq bins adn estimated months
    output: curves in months
    '''
    para_all = pd.read_csv('Final_scores/Figures/extrapolation/by_freq/exp.csv')
    
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
    fig_dir = 'Final_scores/Figures/extrapolation/by_freq/' 
    if not os.path.exists(fig_dir):
         os.makedirs(fig_dir)
    fig_path = fig_dir + lang + '_' + vocab_type +  '_sensitivity.png'
    plt.savefig(fig_path, dpi=800)
    plt.show()
    plt.clf()
    

def main(argv):
    # Args parser
    args = parseArgs(argv)
    accum_threshold = args.accum_threshold
    exp_threshold = args.exp_threshold
    color_dict = args.color_dict
    
    para_all = pd.DataFrame()
    for vocab_type in args.vocab_type_lst:
        for lang in args.lang_lst:
            for test_set in args.testset_lst:

                model_dir = args.model_dir + lang + \
                            '/' + vocab_type + '/'
                human_dir = args.human_dir + lang + '/' + vocab_type
                
                if args.aggregate_freq:
                    
                    aggre_freq(vocab_type,human_dir,model_dir,test_set,accum_threshold,exp_threshold
                                   ,lang,args.extrapolation,args.target_y,color_dict)
                    
                elif args.by_freq:
                    
                    set_lst = ['human','CHILDES','unprompted','prompted']
                    
                    if args.extrapolation:
                        para_frame = pd.DataFrame()
                        for set_type in set_lst:
                            
                            para_set = plot_by_freq(vocab_type,human_dir,model_dir,test_set,accum_threshold,exp_threshold
                                         ,lang,set_type,args.extrapolation,args.target_y,color_dict)
                            
                            para_set['set'] = set_type
                            
                            para_frame = pd.concat([para_frame,para_set])
                        para_frame['lang'] = lang
                        para_all = pd.concat([para_all,para_frame])
                        para_all.to_csv('Final_scores/Figures/extrapolation/by_freq/exp.csv' )
                        
                    else:
                        for set_type in set_lst:
                            para_set = plot_by_freq(vocab_type,human_dir,model_dir,test_set,accum_threshold,exp_threshold
                                         ,lang,set_type,args.extrapolation,args.target_y,color_dict)
                elif args.freq_analysis:
                    freq_sensitivity(lang, vocab_type,color_dict)
                    
                else:
                    para_frame = plot_all(vocab_type, human_dir,model_dir, test_set, accum_threshold
                         ,exp_threshold,lang,args.extrapolation,color_dict,args.target_y) 
                    
                    freq = 'avg'
                    para_frame['lang'] = lang
                    para_frame['freq'] = freq
                    para_all = pd.concat([para_all,para_frame])
                    
                
                
                
                        
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)