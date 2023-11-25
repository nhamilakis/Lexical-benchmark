#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot longitudianl change of child-parent averaged utterance entropy and prob
-> normalized by the utterance length 
@author: jliu
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

data = pd.read_csv('Prompt_AE.csv')

def plot_all(data,lang,data_type):
    
    def plot_prob(data,speaker,data_type):
        
        if data_type == 'entropy':
            header = speaker + '_h' 
        elif data_type == 'log prob':
            header = speaker + '_prob'
            
        data_frame = pd.DataFrame([data['month'].tolist(), data[header].tolist()]).T
        frame_by_month = data_frame.rename(columns={0: "month",1: header})
        sns.set_style('whitegrid') 
        
        ax = sns.lineplot(x="month", y=header, data=frame_by_month, label = header)
        plt.title(lang + ': ' + data_type, fontsize=20)
        # set the limits of the x-axis for each line
        for line in ax.lines:
            plt.xlim(0,36)
            plt.ylim(5,15)
        plt.ylabel(data_type)
        plt.savefig(lang + '_' + data_type + '.png', dpi=300, bbox_inches='tight') 

    plot_prob(data,'child',data_type)
    plot_prob(data,'parent', data_type)


plot_all(data,'American English','entropy')
plot_all(data,'American English','log prob')