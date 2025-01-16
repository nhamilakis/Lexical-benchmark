"""Build script for CHILDES-realistic variations.
split out the dev set
"""
from pathlib import Path
from lexical_benchmark import settings
from tqdm import tqdm


file_mode = 'test'
lang = "EN"
root_dir: Path = settings.PATH.dataset_root / "STELATranscriptions2"
dev_loc = root_dir / file_mode / lang 
train_loc = root_dir / "by_month" / lang 
filename = 'transcription.txt'
out_filename = 'char.txt'

# select and concatenate the file


text_lst = []
if file_mode in ['dev','test']:
    data_location = dev_loc
    with open(data_location/filename,'r') as f:
        sentences = f.readlines()
        for sent in sentences:
            text = format_text_with_boundaries(sent)
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
                    text = format_text_with_boundaries(sent)
                    text_lst.append(text)
            with open(file.parent /out_filename,'w') as f:
                for text in text_lst:
                    f.write(text + '\n')
            print(f'Finished writing file to {file.parent /out_filename}')

