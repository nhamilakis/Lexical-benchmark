"""phonemize one transcript for training"""
import argparse
import re
import string
import sys
from pathlib import Path
from tqdm import tqdm
from dp.phonemizer import Phonemizer



def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='preprocess dataset')

    parser.add_argument('--raw_path', type=str,
                        default='/scratch1/projects/lexical-benchmark/datasets/data/machine/train/EN/50h/00/',
                        help='path to raw transcription')
    parser.add_argument('--out_path', type=str,
                        default='/scratch2/jliu/freq_bias_benchmark/data/',
                        help='path to output')
    parser.add_argument('--phonemizer_path', type=str,
                        default='en_us_cmudict_ipa_forward.pt', help='path to phonemizer')
    parser.add_argument('--debug', default=True,
                        help='if debug, only do the first 2 lines')
    parser.add_argument('--preserve_sequences', default=["iː", "uː", 'ɝː', 'ɑː',"oʊ", 'aɪ', 'eɪ', 'dʒ', 'aʊ', 'tʃ', 'ɔɪ'],
                        help='preserved seq for phone segmentation')
    return parser.parse_args(argv)


def clean_txt(sent:str)->str:
    """clean the input string"""
    # Filter out non-ASCII characters
    sent = ''.join(char for char in sent if ord(char) < 128)
    # remove punctuations
    translator = str.maketrans('', '', string.punctuation + string.digits)
    translator[ord('-')] = ' '  # Replace hyphen with blank space
    clean_string = sent.translate(translator).lower()  # lower results
    clean_string = re.sub(r'\s+', ' ', clean_string) # remove redundent blank
    clean_string = clean_string.strip() # remove initial blank
    return clean_string

def phonemize(sent:str,preserve_sequences:list,phonemizer):
    # Escape special characters in preserve_sequences to avoid regex issues
    escaped_sequences = [re.escape(seq) for seq in preserve_sequences]

    # Create a regex pattern to match the sequences that should be preserved
    preserve_pattern = '|'.join(escaped_sequences)

    # Initialize the final strings
    processed_phon_with = ''
    processed_phon_without = ''

    # Function to segment the word
    def segment_word(word):
        # Split the word by preserved sequences
        parts = re.split(f'({preserve_pattern})', word)
        # Remove any empty strings from the list
        parts = [part for part in parts if part]
        segmented = []
        for part in parts:
            if re.fullmatch(preserve_pattern, part):
                # If the part matches a preserved sequence, add it directly
                segmented.append(part)
            else:
                # Otherwise, split the part into individual characters
                segmented.extend(part)
        return segmented

    # Split the input string into words
    phon_string = phonemizer(sent, lang='en_us')
    words = phon_string.split()
    for word in words:
        segmented_word = segment_word(word)
        processed_phon_with += ' '.join(segmented_word) + " | "
        processed_phon_without += ' '.join(segmented_word) + " "

    # Strip the trailing separators and remove leading/trailing blanks
    processed_phon_with = processed_phon_with.rstrip(" | ").strip()
    processed_phon_without = processed_phon_without.rstrip(" ").strip()

    return processed_phon_with, processed_phon_without




def preprocess(raw:list,preserve_sequences:list,phonemizer,debug):
    '''
    input: the string list
    return: the cleaned files
    '''
    raw = [line.strip() for line in raw if line.strip()]
    if debug:     # only phonemize the first 2 line s
        raw = raw[:2]
    processed_without_phon_all = []
    processed_with_phon_all = []
    processed_without_all = []
    processed_with_all = []
    sent_all = []
    for sent in tqdm(raw):
        clean_string = clean_txt(sent)
        processed_phon_without,processed_phon_with = phonemize(clean_string,preserve_sequences,phonemizer)
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
        processed_without_phon_all.append(processed_phon_without)
        processed_with_phon_all.append(processed_phon_with)

    # convert the final results into
    return sent_all,processed_with_all, processed_without_all,processed_with_phon_all, processed_without_phon_all


def write_files(raw_path:Path,out_path:Path,preserve_sequences:list,phonemizer,debug):
    """write all the processed files fom a single raw file"""

    with open(raw_path/'transcription.txt', 'r') as f:
        print(f"Loading raw file from {str(raw_path)}")
        raw = f.readlines()
        cleaned, char_with, char_without,phon_with, phon_without = preprocess(raw,preserve_sequences,phonemizer,debug)

    # wrtie out the results
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / 'cleaned.txt', 'w') as f:
        # Write each element of the list to the file
        for item in cleaned:
            f.write('%s\n' % item)
        print(f"Writing cleaned file to {str(out_path / 'cleaned.txt')}")

    with open(out_path / 'char_with.txt', 'w') as f:
        # Write each element of the list to the file
        for item in char_with:
            f.write('%s\n' % item)
        print(f"Writing cleaned file to {str(out_path / 'char_with.txt')}")

    with open(out_path / 'char_without.txt', 'w') as f:
        # Write each element of the list to the file
        for item in char_without:
            f.write('%s\n' % item)
        print(f"Writing cleaned file to {str(out_path / 'char_without.txt')}")

    with open(out_path / 'phon_with.txt', 'w') as f:
        # Write each element of the list to the file
        for item in phon_with:
            f.write('%s\n' % item)
        print(f"Writing cleaned file to {str(out_path / 'phon_with.txt')}")

    with open(out_path / 'phon_without.txt', 'w') as f:
        # Write each element of the list to the file
        for item in phon_without:
            f.write('%s\n' % item)
        print(f"Writing cleaned file to {str(out_path / 'phon_without.txt')}")




def main(argv):
    # load args
    args = parseArgs(argv)
    raw_path = Path(args.raw_path)
    out_path = Path(args.out_path)
    preserve_sequences = args.preserve_sequences
    phonemizer_path = 'en_us_cmudict_ipa_forward.pt'
    phonemizer = Phonemizer.from_checkpoint(phonemizer_path)
    print(f'Have loaded phonemizer from {phonemizer_path}')

    # loop different chunks
    for chunk in raw_path.iterdir():
        if chunk.is_dir():
            print(f'Loading files from {chunk}')
            write_files(chunk,chunk,preserve_sequences,phonemizer,args.debug)
            print('########################')
            print(f'Finished preprocessing files from {chunk}')





if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)

