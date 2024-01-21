# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:14:06 2023

@author: 12631
"""

import json
import argparse

import torch


from fairseq import tasks, checkpoint_utils


def readArgs(pathArgs):
    print(f"Loading args from {pathArgs}")
    with open(pathArgs, 'r') as file:
        args = argparse.Namespace(**json.load(file))
    return args

def writeArgs(pathArgs, args):
    print(f"Writing args to {pathArgs}")
    with open(pathArgs, 'w') as file:
        json.dump(vars(args), file, indent=2)




def loadLSTMLMCheckpoint(pathLSTMCheckpoint, pathData):
    """
    Load lstm_lm model from checkpoint.
    """
    # Set up the args Namespace
    model_args = argparse.Namespace(
        task='language_modeling',
        output_dictionary_size=-1,
        data=pathData,
        path=pathLSTMCheckpoint
        )

    # Setup task
    task = tasks.setup_task(model_args)
    
    # Load model
    models, _model_args = checkpoint_utils.load_model_ensemble([model_args.path], task=task)
    model = models[0]
    
    return model, task