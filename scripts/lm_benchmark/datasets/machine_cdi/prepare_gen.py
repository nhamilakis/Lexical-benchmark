import pandas as pd

model_dict = {'50h':[1],'100h':[1],'200h':[2,3],'400h':[4,8],'800h':[9,18]
    ,'1600h':[19,28],'3200h':[29,36],'4500h':[46,54],'7100h':[66,74]}


model_dict = {'50h':[1],'100h':[1],'200h':[2,3],'400h':[4,8],'800h':[10,18]
    ,'1600h':[19,28],'3200h':[29,36],'4500h':[46,54],'7100h':[66,74]}

df = pd.read_csv('/Users/jliu/PycharmProjects/Lexical-benchmark/datasets/raw/CHILDES_child.csv'
                   ,usecols = ['month','content','num_tokens'])

month_lst = ['3200h']
for month in month_lst:
    selected = df[(df['month'] >= model_dict[month][0]) & (df['month'] <= model_dict[month][1])]
    selected.columns = ['train','month','num_tokens']
    selected.to_csv('/Users/jliu/PycharmProjects/freq_bias_benchmark/data/train/train_utt/' + month[:-1] + '_child1.csv')


# segment the dataframes
import os
import numpy as np

month = '800h'
# Load the DataFrame from the CSV file
df = pd.read_csv('/Users/jliu/PycharmProjects/freq_bias_benchmark/data/train/train_utt/' + month[:-1] + '.csv')

# Define the number of splits and directories
num_splits = 30
num_dirs = 3
files_per_dir = num_splits // num_dirs

# Calculate the size of each split
split_size = int(np.ceil(len(df) / num_splits))

# Create directories
for i in range(1, num_dirs + 1):
    os.makedirs(f'dir_{i}', exist_ok=True)

# Split the DataFrame and save to files
for i in range(num_splits):
    start_idx = i * split_size
    end_idx = min((i + 1) * split_size, len(df))
    split_df = df.iloc[start_idx:end_idx]

    # Determine the directory for the current file
    dir_idx = (i // files_per_dir) + 1
    file_name = f'dir_{dir_idx}/data_{i + 1}.csv'

    # Save the split DataFrame to a CSV file
    split_df.to_csv(file_name, index=False)
