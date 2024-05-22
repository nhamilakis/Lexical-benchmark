#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random



def d_stats(x):
    """"descriptive stats for an array of values"""
    stats={'mean':np.mean(x),
           'median':np.median(x),
           'min':np.min(x),
           'max':np.max(x),
           'stdev':np.std(x,ddof=1),
#           'count':len(x),
           'first':np.percentile(x, 25),
           'third':np.percentile(x, 75)}
    return stats


def bin_stats(x, N):
    """Divide the array x into N bins and compute stats for each."""
    # Sort the array
    x_sorted = np.sort(x)
    # Calculate the number of elements in each bin
    n = len(x_sorted) // N
    bins = [x_sorted[i:i + n] for i in range(0, len(x_sorted), n)]
    # Ensure we use all elements (important if len(x) is not perfectly divisible by N)
    if len(x_sorted) % N:
        bins[-2] = np.concatenate((bins[-2], bins[-1]))
        bins.pop()
    # Compute stats for each bin using get_stats
    stats_list = [d_stats(bin) for bin in bins]
    # Create DataFrame from the list of stats dictionaries
    df = pd.DataFrame(stats_list)
    return df


def loss(refstats, teststats):
    """L2 norm of the difference between refstats and teststats"""
    return np.sum(np.sum((refstats-teststats)**2))

def init_index(N,P):
   """returns two indexes, one for positives, one for negatives"""
   idx = np.arange(N)
   result_list = [True] * P + [False] * (N - P)

   # Shuffle the list to mix the Trues and Falses randomly
   random.shuffle(result_list)
   result_array = np.array(result_list)
   return idx[result_array],idx[np.logical_not(result_array)]

def swap_index(pidx,nidx):
    """Randomly swap an element from pidx and nidx"""
    i=random.randint(0, len(pidx)-1)
    j=random.randint(0, len(nidx)-1)
    p1=np.array(pidx,copy=True)
    n1=np.array(nidx,copy=True)
    p1[i], n1[j] = n1[j], p1[i]
    return p1,n1

def get_freq(dataref,header):
    return np.log10(dataref[header]/dataref[header].sum()*1000000)