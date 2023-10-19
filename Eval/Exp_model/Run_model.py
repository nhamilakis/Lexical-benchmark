# -*- coding: utf-8 -*-
"""
Run models recursively

"""

import subprocess


def run_command(command):
    subprocess.call(command, shell=True)



preprocess_command = 'fairseq-preprocess --only-source --task masked_lm \
  --trainpref /scratch2/jliu/STELAWord/data/preprocessed/EN/hour/00/All_with_train.txt \
  --validpref /scratch2/jliu/STELAWord/data/preprocessed/EN/hour/00/All_with_val.txt \
  --destdir /scratch2/jliu/STELAWord/data/preprocessed/EN/hour/00/bin_with \
  --workers 20'
  
  
model_command = 'fairseq-train --task language_modeling \
  /scratch2/jliu/STELAWord/data/preprocessed/EN/hour/00/bin_with \
  --save-dir /scratch2/jliu/STELAWord/models/char_with/hour/00/checkpoints \
  --tensorboard-logdir /scratch2/jliu/STELAWord/models/char_with/hour/00/tensorboard \
  --fp16 \
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
  --log-file /scratch2/jliu/STELAWord/models/char_with/hour/00/result.log \
  --max-tokens 24576 \
  --update-freq 16 \
  --max-update 50000'



hour_lst = ['100h','1600h','200h', '3200h','400h', '50h',  '800h']
# recursively run the command
for hour_name in hour_lst:
    
    preprocess_command = preprocess_command.replace('hour', hour_name)
    run_command(preprocess_command)
    model_command = model_command.replace('hour', hour_name)
    run_command(model_command)
        


