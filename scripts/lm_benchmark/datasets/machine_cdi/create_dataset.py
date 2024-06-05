"""
Construct same-sized datasets in/out of the domain
"""
import os
import re
import string
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment


ROOT = "/Users/jliu/PycharmProjects/Lexical-benchmark"

def load_metadata(meta_data_path:str,text_dir:str,out_dir:str)->pd.DataFrame:
    """load metadata for a fine-grained match"""
    meta_data = pd.read_csv(meta_data_path)
    def get_title(path_name: str) -> str:
        return path_name.split('/')[-1]
    meta_data['filename'] = meta_data['text_path'].apply(get_title)

    all_file_lst = os.listdir(text_dir)
    print(f'Counting token_num from {text_dir}')
    selected_data = meta_data[meta_data['filename'].isin(all_file_lst)]
    num_tokens = []
    for file in selected_data['filename'].tolist():
        frame = txt2csv(text_dir, file)
        num_tokens.append(frame['num_tokens'].tolist()[0])
    selected_data['num_tokens'] = num_tokens
    selected_data.to_csv(out_dir)
    return selected_data


def clean_text(loaded:list):
    """
    remove digits and punct of a text string
    Returns
    -------
    a list of the cleaned string
    """
    result = [line for line in loaded if line.strip()]
    cleaned_text = []
    for sent in tqdm(result):
        # Filter out non-ASCII characters
        sent = ''.join(char for char in sent if ord(char) < 128)
        # remove punctuations
        translator = str.maketrans('', '', string.punctuation + string.digits)
        translator[ord('-')] = ' '  # Replace hyphen with blank space
        clean_string = sent.translate(translator).lower()
        clean_string = re.sub(r'\s+', ' ', clean_string)
        clean_string = clean_string.strip()
        cleaned_text.append(clean_string)
    return cleaned_text


def count_token(text):
    return len(text.split())

def txt2csv(text_dir:str, txt:str):
    """convert txt file into csv dataframe: filename\ train\ num_token """
    # read train filename
    with open(text_dir + txt, encoding="utf8") as f:
        lines = f.readlines()
        cleaned_lines = clean_text(lines)
        frame = pd.DataFrame(cleaned_lines)
        # assign column headers
        frame = frame.rename(columns={0: 'train'})
        frame['num_tokens'] = frame['train'].apply(count_token)
        frame.insert(loc=0, column='filename', value=txt)
    return frame



def remove_file(large_list,sublist_to_remove):
    # Find the index where the sublist to remove starts
    start_index = large_list.index(sublist_to_remove[0])
    # Remove the sublist from the large list
    large_list = large_list[:start_index] + large_list[start_index + len(sublist_to_remove):]
    return large_list




def cut_df(df,target_cum_sum,header = 'num_tokens'):
    """cut df rows until it has reached the target value"""
    # Calculate cumulative sum
    cum_sum = df[header].cumsum()
    # Find the index where the cumulative sum exceeds or equals the target value
    index_to_cut = cum_sum[cum_sum >= target_cum_sum].index.min()
    # If no index found, keep all rows
    if pd.isnull(index_to_cut):
        index_to_cut = len(df)
    # Remove rows after the index_to_cut
    df = df.iloc[:index_to_cut]
    return df


def match_dataframes(dfA, dfB):
    """match files based on """

    matched_rows = []
    for genre in dfA['genre'].unique():
        dfA_genre = dfA[dfA['genre'] == genre]
        dfB_genre = dfB[dfB['genre'] == genre]

        if len(dfB_genre) < len(dfA_genre):
            raise ValueError(f"Not enough rows in dfB to match genre '{genre}' in dfA")

        cost_matrix = np.abs(dfA_genre['num_tokens'].values[:, None] - dfB_genre['num_tokens'].values)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_rows.append(dfB_genre.iloc[col_ind])

    matched_dfB = pd.concat(matched_rows)
    return matched_dfB

def get_ind_mat(filename_path:str, train_freq_dir:str, file:str,text_dir:str,meta_data):

    """
    construct the target pseudo dataset to estimate oov token freq
    """
    # read train filename
    file_lst = pd.read_csv(filename_path,header=None)[0].tolist()
    # remove the ones that are already in the train list
    all_file_lst = os.listdir(text_dir)
    candi_lst = remove_file(all_file_lst,file_lst)
    # match the genre
    genre_candi = meta_data[meta_data['filename'].isin(candi_lst)]
    genre_target = meta_data[meta_data['filename'].isin(file_lst)]
    # count token numbers
    matched_df = match_dataframes(genre_target, genre_candi)
    # get the total number of tokens
    train_num = pd.read_csv(train_freq_dir + file)['num_tokens'].sum()

    train_frame = pd.DataFrame()
    # get constructed set
    for file in matched_df['filename'].tolist():
        frame = txt2csv(text_dir, file)
        train_frame = pd.concat([train_frame,frame])
    return train_frame, train_num




def get_ood_mat(text_path:str, train_freq_dir:str, out_dir:str):

    """
    construct the target pseudo dataset from CHIDLES transcript
    """
    # get constructed set
    frame = pd.read_csv(text_path)
    frame = frame.dropna()
    frame = frame.sample(frac=1, random_state=66).reset_index(drop=True)
    # loop train_freq file
    for file in tqdm(os.listdir(train_freq_dir)):
        try:
            # count token numbers
            train_num = pd.read_csv(train_freq_dir + file)['num_tokens'].sum()
            oov_sum = 0
            train_frame = pd.DataFrame()
            n = 0
            while oov_sum < train_num:
                oov_sum += frame['num_tokens'].sum()
                train_frame = pd.concat([train_frame,frame])
                n += 1

            # cut additional line to align with the target train set
            train_frame = cut_df(train_frame,train_num,'num_tokens')
            # print out the utt
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            train_frame.to_csv(out_dir + file)

        except:
            print(file)


def main():

    mode = 'ood'
    # filenames of the largest set to remove all the possible files
    train_freq_dir = '/Users/jliu/PycharmProjects/Lexical-benchmark/datasets/raw/train/'
    out_dir = '/Users/jliu/PycharmProjects/Lexical-benchmark/datasets/raw/' + mode + '/'

    if mode == 'ind':
        meta_data_path = f"{ROOT}/datasets/raw"
        text_dir = '/Users/jliu/PycharmProjects/Lexical-benchmark/datasets/raw/audiobook/'
        if not os.path.exists(meta_data_path + "/matched.csv"):
            print('No metadata available, creating...')
            meta_data = load_metadata(meta_data_path + "/matched2.csv", text_dir, meta_data_path + "/matched.csv")
        else:
            meta_data = pd.read_csv(meta_data_path + "/matched.csv")
        filename_path = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/train/filename/7100.csv'
        
        file = '400.csv'
        train_frame, train_num = get_ind_mat(filename_path, train_freq_dir, file, text_dir,meta_data)

        # print out the utt
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        train_frame.to_csv(out_dir + file)
        train_frame = pd.read_csv(out_dir + file)
        train_frame = cut_df(train_frame, train_num)
        train_frame.to_csv(out_dir + file)


    else:
        text_path = '/Users/jliu/PycharmProjects/Lexical-benchmark/datasets/raw/CHILDES_adult.csv'
        get_ood_mat(text_path, train_freq_dir, out_dir)

if __name__ == "__main__":
    main()


