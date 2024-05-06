import pandas as pd
import collections

def get_freq_table(result):
    """get freq from a word list"""
    # clean text and get the freq table
    frequencyDict = collections.Counter(result)
    freq_lst = list(frequencyDict.values())
    word_lst = list(frequencyDict.keys())

    # get freq
    fre_table = pd.DataFrame([word_lst, freq_lst]).T
    col_Names = ["word", "count"]
    fre_table.columns = col_Names
    fre_table['freq_m'] = fre_table['count'] / len(result) * 1000000
    return fre_table
