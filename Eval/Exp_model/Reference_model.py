# -*- coding: utf-8 -*-
"""
train the reference model on opensubtitle etc recursively
idea: 10,000 words/h * 1,000h * 3 = 30,000,000 words  in total

input: Training dataset and months to be simulated
output: trained model checkpoints
"""

import os
import subprocess
import argparse
import sys
 



def parseArgs(argv):
    # Run parameters: we have similar folder structure, so juat input lang
    parser = argparse.ArgumentParser(description='train reference models')
    
    parser.add_argument('--month', type=str, default = '36',
                        help='the target month')
    
    parser.add_argument('--TextPath', type=str, default = '/scratch2/jliu/STELAWord/productive_vocab/reference/data/',
                        help='rootpath of the text')
    
    
    return parser.parse_args(argv)



def run_command(command):
    subprocess.call(command, shell=True)


def get_data(TextPath, month):
    
    '''
    step1: 
        get the number of words in different months
        
    step2:
        generate and put in the corresponding folders
    '''
    
    
    text_len = int(month) * 30 * 10000       
    
    with open(TextPath + '36/train.txt', 'r', encoding="utf8", errors='ignore') as file:
        combined_text = file.read()
        
    selected_text = ''.join(combined_text.split()[:text_len])
    
    if not os.path.exists(TextPath + month):  
        os.mkdir(TextPath + month)
    # get the target length of the text  
    with open(TextPath + month + '/train.txt', 'w', encoding="utf8", errors='ignore') as file:
        file.write(selected_text)




def main(argv):
    
    # Args parser
    args = parseArgs(argv)
    
    month = args.month
    TextPath = args.TextPath
    
   
    # step 1: load data
    get_data(TextPath, month)
    
    
    # step 2: calculate the results
    
    # first tokenize and get the bin data
    preprocess_command = 'fairseq-preprocess --only-source --task masked_lm \
      --trainpref /scratch2/jliu/STELAWord/productive_vocab/reference/data/36/train.txt \
      --validpref /scratch2/jliu/STELAWord/productive_vocab/reference/data/36/val.txt \
      --destdir /scratch2/jliu/STELAWord/productive_vocab/reference/data/36/bin_data \
      --workers 20'
      

    # train the model
    train_command = 'fairseq-train --task language_modeling \
      /scratch2/jliu/STELAWord/productive_vocab/reference/data/36/bin_data \
      --save-dir /scratch2/jliu/STELAWord/productive_vocab/reference/model/36/checkpoints \
      --tensorboard-logdir /scratch2/jliu/STELAWord/productive_vocab/reference/model/36/checkpoints/tensorboard \
      --arch lstm_lm \
      --keep-last-epochs 2 \
      --patience 5 \
      --dropout 0.1 \
      --weight-decay 0.01 \
      --decoder-embed-dim 200 \
      --decoder-hidden-size 1024 \
      --decoder-layers 3 \
      --decoder-out-embed-dim 200 \
      --optimizer adam \
      --adam-betas \'(0.9, 0.98)\' \
      --clip-norm 0.0 \
      --lr-scheduler inverse_sqrt \
      --lr 0.01 \
      --warmup-updates 4000 \
      --warmup-init-lr 1e-07 \
      --tokens-per-sample 2048 \
      --sample-break-mode none \
      --add-bos-token \
      --log-format simple \
      --log-interval 1 \
      --log-file /scratch2/jliu/STELAWord/productive_vocab/reference/model/36/checkpoints/result.log \
      --max-tokens 24576 \
      --update-freq 16 \
      --max-update 50000'

    preprocess_command = preprocess_command.replace('36/train', month + '/train')
    preprocess_command = preprocess_command.replace('36/bin', month + '/bin')
    train_command = train_command.replace('36', month)
    run_command(preprocess_command)
    run_command(train_command)
    
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)

  


