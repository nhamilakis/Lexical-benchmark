"""prepare for materials of spelling checker"""

import pandas as pd


def extract_word_list_with_pandas(file_path, chunksize=10000):
    word_list = []
    for chunk in pd.read_json(file_path, lines=True, chunksize=chunksize):
        words = chunk['word'].dropna().tolist()
        word_list.extend(words)
    unique_words = set([word.lower() for word in word_list])
    return unique_words

ROOT = "/Users/jliu/PycharmProjects/Lexical-benchmark/datasets/raw"

CELEX = pd.read_excel(f'{ROOT}/SUBTLEX.xlsx')['Word'].str.lower().tolist()
words = extract_word_list_with_pandas(f'{ROOT}/wiki.json')
# prepare for the word list
intersection = CELEX.intersection(words)

