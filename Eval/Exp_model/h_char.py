#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract sequence probability and entropy from CANINE-c

@author: jliu
"""
from transformers import CanineModel,CanineTokenizer
import torch

tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
model = CanineModel.from_pretrained("google/canine-c")
# get the mask token index
vocab_dict = tokenizer.get_vocab()
masked_idx = vocab_dict[""]     # the mask token comes fromthe source code 

# Compute the input sequences of tokens
masked_sentence_tokens_list = []
sequences_list = []         # to retrieve the sentences when computing logproba
sentences = ["hello world"]

# tokenize the input sequences
for sentence in sentences:
    sentence_tokens = tokenizer.encode(sentence, padding="longest", truncation=True, return_tensors="pt")
    ln = len(sentence_tokens) - 2    

    # Start masking out tokens
    for idx in range(0, 1, ln):      #!!! why we need the step_size in the code?
        masked_sentence_tokens = sentence_tokens.clone().long()
        masked_sentence_tokens[idx+1:idx+1+ln] = masked_idx
        masked_sentence_tokens_list.append(masked_sentence_tokens.clone())
    sequences_list.append(sentence_tokens)