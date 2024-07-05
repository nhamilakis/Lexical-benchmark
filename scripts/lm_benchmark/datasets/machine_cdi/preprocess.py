import argparse
import os
import re
import string
import sys
import pandas as pd
from tqdm import tqdm

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Select test sets by freq')

    parser.add_argument('--filename_path', type=str,
                        default='/Users/jliu/PycharmProjects/freq_bias_benchmark/data/train/filename/',
                        help='path to corresponding filenames to each model')
    parser.add_argument('--dataset_path', type=str,
                        default='/Users/jliu/PycharmProjects/Machine_CDI/Lexical-benchmark_data/train_phoneme/dataset/',
                        help='path to raw dataset')
    parser.add_argument('--mat_path', type=str,
                        default='/Users/jliu/PycharmProjects/freq_bias_benchmark/data/train/train_mat/',
                        help='path to preprocessed files')
    parser.add_argument('--utt_path', type=str,
                        default='/Users/jliu/PycharmProjects/freq_bias_benchmark/data/train/train_utt/',
                        help='path to preprocessed files')

    return parser.parse_args(argv)


# preprocess the data
def remove_blank(x):
    if isinstance(x, str):  # Check if x is a string
        return x.strip()    # Strip leading and trailing spaces if it's a string

def get_len(x):
    try:
        return len(x)
    except:
        return 0

def count_words(sentence):
    # Split the sentence into words based on whitespace
    words = sentence.split()
    return len(words)

def clean_txt(sent:str):
    """clean the input string"""
    # Filter out non-ASCII characters
    sent = ''.join(char for char in sent if ord(char) < 128)
    # remove punctuations
    translator = str.maketrans('', '', string.punctuation + string.digits)
    translator[ord('-')] = ' '  # Replace hyphen with blank space
    clean_string = sent.translate(translator).lower()
    clean_string = re.sub(r'\s+', ' ', clean_string)
    clean_string = clean_string.strip()
    return clean_string



def preprocess(raw:list):
    '''
    input: the string list
    return: the cleaned files
    '''
    raw = [line.strip() for line in raw if line.strip()]
    processed_without_all = []
    processed_with_all = []
    sent_all = []
    for sent in tqdm(raw):
        clean_string = clean_txt(sent)
        word_lst = clean_string.split(' ')
        # convert into corresponding format string
        processed_with = ''
        processed_without = ''
        for word in word_lst:
            upper_char = ' '.join(word).upper()
            if not word.isspace():
                processed_with += upper_char + " | "
                processed_without += upper_char + " "

        sent_all.append(clean_string)
        processed_without_all.append(processed_without)
        processed_with_all.append(processed_with)
    # convert the final results into
    return sent_all,processed_with_all, processed_without_all



def get_utt_frame(sent:list,file:str,all_frame):
    """save the cleaned utt as a dataframe"""
    utt_frame = pd.DataFrame(sent)
    utt_frame = utt_frame.rename(columns={0: 'train'})
    utt_frame['filename'] = file
    all_frame = pd.concat([all_frame, utt_frame])
    return all_frame



def main(argv):
    # load args
    args = parseArgs(argv)
    filename_path = args.filename_path
    dataset_path = args.dataset_path
    mat_path = args.mat_path
    utt_path = args.utt_path
    if not os.path.exists(mat_path):
        os.makedirs(mat_path)

    if not os.path.exists(utt_path):
        os.makedirs(utt_path)

    for file in os.listdir(filename_path):
        # loop over different hours
        if file.endswith('.csv'):
            # load training data based on filename list
            file_lst = pd.read_csv(filename_path + file,header=None)
            all_frame = pd.DataFrame()
            train = []
            for txt in file_lst[0]:
                with open(dataset_path + txt, 'r') as f:
                    raw = f.readlines()
                    sent_all,processed_with, _ = preprocess(raw)
                    all_frame = get_utt_frame(sent_all,txt,all_frame)
                    train.extend(processed_with)

            # save the utt csv file
            all_frame.to_csv(utt_path + file)
            print(f"Finish prepare utt for {file}")

            out_path = mat_path + file[:-4]
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            # Open the file in write mode
            with open(out_path + '/data.txt', 'w') as f:
                # Write each element of the list to the file
                for item in train:
                    f.write('%s\n' % item)

            print(f"Finish preprocessing {file}")





if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)

