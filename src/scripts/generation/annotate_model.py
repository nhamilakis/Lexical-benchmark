
"""
Prepare the age-relevant generation file
"""
import pandas as pd
from pathlib import Path
from lexical_benchmark import settings
from typing import List, Dict
import numpy as np

# veraibles
target_months = [6, 12, 18, 24, 30, 36]
out_dir: Path = settings.PATH.DATA_DIR / "gen"


def get_prop(target_months:list):
    # return the dictionary with the proportion of different generations
    month_dict = {}
    # loop over the different elements
    n = 0 
    while n < len(target_months):
        if n == 0:
            # take all the prop of 6th month
            for i in range(1,target_months[n]+1):
                month_dict[i] = [{target_months[n]:1}]
        else:
            for i in range(target_months[n-1],target_months[n]+1):
                # always take the right boundary of the month interval
                month_diff_prop = 1- (i - target_months[n-1])/(target_months[n]-target_months[n-1])
                month_dict[i] = [{target_months[n-1]:month_diff_prop},{target_months[n]:1-month_diff_prop}]
        n+=1
    return month_dict


def append_model(
    df: pd.DataFrame,
    proportion_list: List[Dict[int, float]],
    word_count_column: str = 'sent_len',
    target_column: str = 'model'
) -> pd.DataFrame:
    """
    Add an annotation column based on cumulative word counts and target proportions.
    
    Args:
        df (pd.DataFrame): Input DataFrame with word count column
        proportion_list (List[Dict[int, float]]): List of dictionaries with value:proportion pairs
        word_count_column (str): Name of the word count column
        target_column (str): Name of the new annotation column
    
    Returns:
        pd.DataFrame: DataFrame with the new annotation column
    
    Example:
        proportion_list = [{18: 0.8333333333333334}, {24: 0.16666666666666663}]
        result_df = add_proportional_annotation(df, proportion_list)
    """
    try:
        # Input validation
        if word_count_column not in df.columns:
            raise ValueError(f"Word count column '{word_count_column}' not found in DataFrame")
        
        # Create a copy of the DataFrame
        result_df = df.copy()
        
        # Sort DataFrame by word count
        result_df = result_df.sort_values(by=word_count_column)
        
        # Calculate cumulative sum of word counts
        total_words = result_df[word_count_column].sum()
        cumsum = result_df[word_count_column].cumsum()
        cumsum_proportions = cumsum / total_words
        
        # Initialize annotation column
        result_df[target_column] = None
        
        # Create cumulative thresholds from proportion list
        thresholds = []
        values = []
        cumulative_prop = 0
        
        for prop_dict in proportion_list:
            for value, proportion in prop_dict.items():
                cumulative_prop += proportion
                thresholds.append(cumulative_prop)
                values.append(value)
        
        # Assign values based on cumulative proportions
        current_threshold_idx = 0
        for idx, row_proportion in enumerate(cumsum_proportions):
            while (current_threshold_idx < len(thresholds) - 1 and 
                   row_proportion > thresholds[current_threshold_idx]):
                current_threshold_idx += 1
            result_df.iloc[idx, result_df.columns.get_loc(target_column)] = values[current_threshold_idx]
        
        return result_df
        
    except Exception as e:
        print(f"Error in add_proportional_annotation: {str(e)}")
        raise



month_dict = get_prop(target_months)
print(month_dict)
data = pd.read_csv(out_dir/'CHILDES.csv')

# annotate based on the prop
data_grouped = data.groupby('month')
data_all = pd.DataFrame()
for month, data_group in data_grouped:
    month_prop = month_dict[month]
    # annotate with months
    df = append_model(data_group,month_prop)
    data_all = pd.concat([data_all,df])


data_all.to_csv(out_dir/'CHILDES_model.csv')