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
    parser.add_argument('--output_path', type=str,
                        default='/scratch2/jliu/freq_bias_benchmark/data/',
                        help='path to output')
    parser.add_argument('--phonemizer_path', type=str,
                        default='en_us_cmudict_ipa_forward.pt', help='path to phonemizer')
    parser.add_argument('--debug', default=True,
                        help='if debug, only do the first 2 lines')
    parser.add_argument('--preserve_sequences', default=["iː", "uː", 'ɝː', "oʊ", 'aɪ', 'eɪ', 'dʒ', 'aʊ', 'tʃ', 'ɔɪ'],
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

def phonemize(sent:str,preserve_sequences:list,phonemizer)->str:
    # creturn the phonemized string seperated by blanks as there are long vowels and diphthongs
    input_string = phonemizer(sent, lang='en_us')
    # Escape special characters in preserve_sequences to avoid regex issues
    escaped_sequences = [re.escape(seq) for seq in preserve_sequences]
    # Create a regex pattern to match the sequences that should be preserved
    preserve_pattern = '|'.join(escaped_sequences)
    # Split the input string by preserving sequences
    segments = re.split(f'({preserve_pattern})', input_string)
    # Process the segments to build the final result
    result = []
    for segment in segments:
        if re.fullmatch(preserve_pattern, segment):
            # If the segment is a preserved sequence, add it directly
            result.append(segment)
        else:
            # For other segments, treat each character separately
            for char in segment:
                if not char.isspace():
                    result.append(char)
    return ' '.join(result)

def preprocess(raw:list,preserve_sequences:list,phonemizer):
    '''
    input: the string list
    return: the cleaned files
    '''
    raw = [line.strip() for line in raw if line.strip()]
    processed_without_phon_all = []
    processed_with_phon_all = []
    processed_without_all = []
    processed_with_all = []
    sent_all = []
    for sent in tqdm(raw):
        clean_string = clean_txt(sent)
        phon_string = phonemize(sent,preserve_sequences,phonemizer)
        word_lst = clean_string.split(' ')
        # convert into corresponding format string
        processed_with = ''
        processed_without = ''
        for word in word_lst:
            upper_char = ' '.join(word).upper()
            if not word.isspace():
                processed_with += upper_char + " | "
                processed_without += upper_char + " "

        phon_lst = phon_string.split(' ')
        processed_phon_with = ''
        processed_phon_without = ''
        for phon in phon_lst:
            phones = ' '.join(phon)
            if not phon.isspace():
                processed_phon_with += phones + " | "
                processed_phon_without += phones + " "

        sent_all.append(clean_string)
        processed_without_all.append(processed_without)
        processed_with_all.append(processed_with)
        processed_without_phon_all.append(processed_phon_without)
        processed_with_phon_all.append(processed_phon_with)

    # convert the final results into
    return sent_all,processed_with_all, processed_without_all,processed_with_phon_all, processed_without_phon_all



def main(argv):
    # load args
    args = parseArgs(argv)
    raw_path = Path(args.raw_path)
    out_path = Path(args.raw_path)
    preserve_sequences = args.preserve_sequences
    phonemizer_path = 'en_us_cmudict_ipa_forward.pt'
    phonemizer = Phonemizer.from_checkpoint(phonemizer_path)
    print(f'Have loaded phonemizer from {phonemizer_path}')

    # clean the transcript
    with open(raw_path/'transcription.txt', 'r') as f:
        print(f"Loading raw file from {str(raw_path)}")
        raw = f.readlines()
        cleaned, char_with, char_without,phon_with, phon_without = preprocess(raw,preserve_sequences,phonemizer)
        if args.debug:
            cleaned = cleaned[:2]
            char_with = char_with[:2]
            char_without = char_without[:2]
            phon_with = phon_with[:2]
            phon_without = phon_without[:2]
    # wrtie out the results
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


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)

