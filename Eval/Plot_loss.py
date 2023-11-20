#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot the loss figures recursively to check the potential underfitting problem

@author: jliu
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
import os


loss_path = 'model/36/checkpoints'
month = '36'
def get_loss(loss_path, month):
    
    '''
    input: loss file directory
    output: the dataframe with epoch, train_loss, val_loss
    '''
    
    loss_text = pd.read_csv(loss_path + '/result.log',header=None)
    # only select the relevent lines
    index_anchor = loss_text.index[loss_text[0] == 'begin validation on "valid" subset'].tolist()
    # Paths to the Fairseq result log file
    train_loss_lst = []
    val_loss_lst = []
    epoch_lst = []
    for index in index_anchor:
        
        try: 
            train_loss = float(re.search(r'loss=([\d.]+)',loss_text.iloc[[index-1]][0].item()).group(1)) 
            epoch = int(re.search(r'epoch\s([\d.]+)',loss_text.iloc[[index-1]][0].item()).group(1)) 
            train_loss_lst.append(train_loss)
            epoch_lst.append(epoch)
            try:
                val_loss = float(re.search(r'best_loss\s([\d.]+)',loss_text.iloc[[index+1]][0].item()).group(1)) 
            except:
                val_loss = float(re.search(r'loss\s([\d.]+)',loss_text.iloc[[index+1]][0].item()).group(1)) 
            val_loss_lst.append(val_loss)
        except:
            pass
    loss_frame = pd.DataFrame([epoch_lst,train_loss_lst,val_loss_lst]).T
    loss_frame.rename(columns={0: 'epoch',1:'train_loss',2:'val_loss'}, inplace=True)
    
    plt.figure(figsize=(8, 6), dpi=800)
    plt.plot(loss_frame['epoch'].tolist(), loss_frame['train_loss'].tolist(), label='Training Loss')
    plt.plot(loss_frame['epoch'].tolist(), loss_frame['val_loss'].tolist(), label='Validation Loss')
    plt.title('Loss Curves of ' + month + ' month(s) data')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # save figure
    plt.savefig(loss_path + '/plot.png')
    return loss_frame



rootpath = 'model'
    
for month in os.listdir(rootpath):
        print('Plotting the model history of ' + month + ' months')
        try:
            get_loss(rootpath + '/' + month + '/checkpoints', month)
        except:
            print(month)
