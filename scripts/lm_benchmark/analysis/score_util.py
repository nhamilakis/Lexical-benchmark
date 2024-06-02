#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path
from lm_benchmark.utils import TokenCount

################################################################################################
# functions to load crf generations
#################################################################################################

def select_model(model_dict, month_range):
    """
    Selects key-value pairs from the dictionary where the given range falls within the min and max of the ranges.

    Parameters:
        model_dict (dict): A dictionary where keys are strings representing ranges and values are lists of integers.
        month_range (list): A list containing two integers representing the range to check.

    Returns:
        dict: A dictionary with key-value pairs that fall within the specified range.
    """
    selected_pairs = {}

    for key, value in model_dict.items():
        min_value = min(value)
        max_value = max(value)

        # Check if the given range [month_range[0], month_range[1]] falls within the key's range [min_value, max_value]
        if month_range[0] <= max_value and month_range[1] >= min_value:
            selected_pairs[key] = value

    return selected_pairs


def segment_df(df: pd.DataFrame, column: str, target_sum: float) -> (pd.DataFrame, pd.DataFrame):
    """
    Segments a DataFrame into two parts by rows such that the cumulative sum of a specific column in the first part
    is equal to the target_sum if possible.

    Parameters:
        df (pd.DataFrame): The DataFrame to segment.
        column (str): The column on which to base the cumulative sum.
        target_sum (float): The target cumulative sum for the first segment.

    Returns:
        tuple: Two DataFrames representing the segmented parts. If an exact match is not possible,
               the first part will be empty.
    """
    current_sum = 0.0
    split_index = 0

    for idx, value in enumerate(df[column]):
        current_sum += value
        if current_sum >= target_sum:
            split_index = idx + 1  # Include the current row in the first segment
            break
        elif current_sum > target_sum:
            # If the current_sum exceeds target_sum, exact match is not possible
            print(f"Exact match for target_sum {target_sum} is not possible.")
            return pd.DataFrame(), df

    df1 = df.iloc[:split_index].reset_index(drop=True)
    df2 = df.iloc[split_index:].reset_index(drop=True)

    return df1, df2

def merge_crf(gen_ROOT:str,model_dict:dict, month_range:list)->pd.DataFrame:      #TODO: debug this func later
    # loop over each model
    selected_pairs = select_model(model_dict, month_range)
    selected_all = pd.DataFrame()
    for model,model_month in selected_pairs.items():
        # load the model
        filename = model +'.csv'
        crp = pd.read_csv(Path(gen_ROOT)/filename)
        # loop over the month range in each model
        for month in range(model_month[0],model_month[1]+1):
            target_count = generation[generation['month']==month]['num_tokens'].sum()
            selected, crp = segment_df(crp, 'count', target_count)
            selected['month'] = month
            selected_all = pd.concat([selected_all,selected])

    # further select the range
    selected_df = selected_all[(selected_all['month']>=month_range[0])&(selected_all['month']<=month_range[1])]
    # convert into TC object
    selected_df['freq_m'] = selected_df['count']/selected_df['count'].sum() * 1000000
    selected_df['correct']=selected_df['word'].apply(is_word)
    return selected_df


################################################################################################
# functions for MonthCounter class
#################################################################################################

def merge_df(merged_df, df2, header: str,month:int,count:False):    # TODO: preserve the month group info
    """Merge freq dataframes on 'word' column"""
    # create the TC object
    if count:
        count_df = TokenCount()
        count_df.df = df2
    else:
        count_df = TokenCount.from_df(df2, header)
    df2 = count_df.df[['word', 'freq_m']]
    # Merge freq dataframes on index 'word'
    merged_df = pd.merge(merged_df, df2, on='word', how='outer')
    # Fill NaN values with 0
    merged_df = merged_df.fillna(0)
    merged_df = merged_df.rename(columns={'freq_m': month})
    return merged_df


def adjust_count(count,est_df,month):
    """adjust the count based on estimation"""
    coeff = est_df[est_df['month']==month]
    return count * 30 * coeff['sec_per_hour'].item() * coeff['hour'].item() * coeff['word_per_sec'].item() / 1000000


def accum_count(df):
    """get accum count from the second column"""
    # Get the first column (preserved)
    first_column = df.iloc[:, 0]
    # Get cumulative count starting from the second column
    cumulative_counts = df.iloc[:, 1:].cumsum(axis=1)
    # Concatenate the first column with the cumulative counts
    result_df = pd.concat([first_column, cumulative_counts], axis=1)
    return result_df

def load_csv(file_path,start_column):
    # Read the CSV file starting from the given column header
    data = pd.read_csv(file_path)
    # Get the index of the start column
    start_column_index = data.columns.get_loc(start_column)
    # Extract the columns starting from the specified column
    selected_data = data.iloc[:, start_column_index:]
    return selected_data



################################################################################################
# MonthCounter class to weight count b yestimation #
#################################################################################################


class MonthCounter:

    """get the monthly info from the concatenated generation/productions"""
    def __init__(self, gen_file: Path, est_file: Path, test_file: Path, count_all_file: Path,
                 count_test_file: Path, header: str, month_range: list, count):

        if not gen_file.is_file():
            raise ValueError(f'Given file ::{gen_file}:: does not exist !!')
        if not est_file.is_file():
            raise ValueError(f'Given file ::{est_file}:: does not exist !!')
        if not test_file.is_file():
            raise ValueError(f'Given file ::{test_file}:: does not exist !!')
        if not count_all_file.is_file():
            self._merged_df = None     # initialize the merged_all as None if it doesn't exist
            print(f'Count corpus does not exist, creating and saving it to {count_all_file}')
        else:
            print(f'Found count corpus from: {count_all_file}, loading ...')
            self._merged_df = load_csv(count_all_file,'word')

        if not count_test_file.is_file():
            self._selected_rows = None    # initialize the merged_all as None if it doesn't exist
            print(f'Test count does not exist, creating and saving it to {count_test_file}')
        else:
            print(f'Found test count from: {count_test_file}, loading ...')
            self._selected_rows = load_csv(count_test_file,'word')

        self._generation_csv_location = gen_file
        self._estimation_csv_location = est_file
        self._all_csv_location = count_all_file
        self._count_filtered_location = count_test_file
        self._test_csv_location = test_file
        self._header = header
        self._month_range = month_range
        self._count = count
        # Call load method to initialize dataframes
        self.load()

    def load(self):
        """ Load the dataset into dataframes """
        generation_df = pd.read_csv(self._generation_csv_location)
        generation_df['month'] = generation_df['month'].astype(int)
        generation_df = generation_df.sort_values('month')

        self._generation_df = generation_df[(generation_df['month'] >= self._month_range[0]) & (generation_df['month'] <= self._month_range[1])]
        self._estimation_df = pd.read_csv(self._estimation_csv_location)
        self._test_df = load_csv(self._test_csv_location,'word')

        return generation_df

    def adjusted_count_all(self):
        """ Match two freq frames """
        # loop over different months

        self._gen_grouped = self._generation_df.groupby('month')
        self._merged_df = pd.DataFrame(columns=['word', 'freq_m'])

        for month, gen_month in self._gen_grouped:
            # get freq in the given month and merge adjusted the count with previous one
            try:
                self._merged_df = merge_df(self._merged_df, gen_month, self._header, month,self._count)
                # try to rename the initial months' header
                self._merged_df = self._merged_df.rename(columns={'freq_m_y': month})
                # adjust count based on estimation
                #self._merged_df[month] = self._merged_df[month].apply(lambda x: adjust_count(x, self._estimation_df, month))
                print('Finished processing ' + str(month))
            except:
                print('Failed processing ' + str(month))

        # remove useless columns
        try:
            self._merged_df = self._merged_df.drop(columns=['freq_m_x'])
        except:
            pass
        # get cumulative frequency
        self._merged_df = accum_count(self._merged_df)
        self._merged_df.to_csv(self._all_csv_location)
        return self._merged_df


    def get_count(self):
        """ Get matched data """
        if self._merged_df is None:
            self._merged_df = self.adjusted_count_all()
        # filter the test set
        self._selected_rows = self._merged_df[self._merged_df['word'].isin(self._test_df['word'])]
        self._selected_rows.to_csv(self._count_filtered_location)




