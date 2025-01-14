
"""Build script for CHILDES-realistic variations.

This script helps build the `txt_merged` folder in the STELA dataset, which creates the same
chunking as the original but its build from the ground up by merging two 50h chunks to create a 100h one, etc..

That way we know that the totals are always the same.
"""
from pathlib import Path
from lexical_benchmark import settings
from tqdm import tqdm


def format_text_with_boundaries(text: str,add_space:True) -> str:
    """
    Format text by:
    1. Removing punctuation 
    2. Adding spaces between characters
    3. Adding boundary markers between words
    
    Args:
        text: Input text string
    Returns:
        Formatted text with character spacing and word boundaries
    """
    # Remove punctuation and split into words
    words = text.split()
    if add_space:
        # Format each word by adding spaces between characters
        spaced_words = [' '.join(word.lower()) for word in words]
        # Join words with boundary marker
        return ' | '.join(spaced_words) + ' |'
    else:
        spaced_words = [''.join(word.lower()) for word in words]
        # Join words with boundary marker
        return '|'.join(spaced_words) + '|'
    

model_type = 'hf'    # no need to add space in huggingface models
file_mode = 'train'
lang = "EN"
root_dir: Path = settings.PATH.dataset_root / "STELATranscriptions2"
dev_loc = root_dir / file_mode / lang 
train_loc = root_dir / "by_month" / lang 
filename = 'transcription.txt'

if model_type == 'hf':
    out_filename = 'char_hf.txt'
    add_space = False
else:
    out_filename = 'char.txt'
    add_space = True


text_lst = []
if file_mode in ['dev','test']:
    data_location = dev_loc
    with open(data_location/filename,'r') as f:
        sentences = f.readlines()
        for sent in sentences:
            text = format_text_with_boundaries(sent,add_space=add_space)
            text_lst.append(text)

    with open(data_location/out_filename,'w') as f:
        for text in text_lst:
            f.write(text + '\n')

    print(f'Finished writing file to {data_location/out_filename}')

elif file_mode == 'train':
    # loop month and chunks
    data_location = train_loc 
    for file in tqdm(data_location.rglob(filename)):  # '*' matches all files and directories
        if file.is_file():  # Check if it's a file (you can also check for directories with is_dir())
            with open(file,'r') as f:
                sentences = f.readlines()
                text_lst = []
                for sent in sentences:
                    text = format_text_with_boundaries(sent,add_space=add_space)
                    text_lst.append(text)
            with open(file.parent /out_filename,'w') as f:
                for text in text_lst:
                    f.write(text + '\n')
            print(f'Finished writing file to {file.parent /out_filename}')

