
"""Build script for CHILDES-realistic variations.

This script helps build the `txt_merged` folder in the STELA dataset, which creates the same
chunking as the original but its build from the ground up by merging two 50h chunks to create a 100h one, etc..

That way we know that the totals are always the same.
"""

#!/home/nhamilakis/envs/venvs/lbenchmark/bin/python3.11
# fmt: off
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=stela-cleanups
#SBATCH --time=5:00:00
#SBATCH --export=ALL
#SBATCH --output stela-clean-%J.log
# fmt: on
"""Build script for STELA/txt_merged.

This script helps build the `txt_merged` folder in the STELA dataset, which creates the same
chunking as the original but its build from the ground up by merging two 50h chunks to create a 100h one, etc..

That way we know that the totals are always the same.

/scratch1/projects/lexical-benchmark/v2/datasets/STELATranscriptions2/txt_merged/EN/50h/00
"""

import os
import pandas as pd
from tqdm import tqdm
from lexical_benchmark import settings
from pathlib import Path



def split_by_count(sentences: list, proportion: float):
    """
    Split a list of sentences into two lists based on the proportion of total word count.

    Parameters:
        sentences (list): A list of sentences.
        proportion (float): The desired proportion for the first sub-list (0 < proportion < 1).

    Returns:
        sub_list1 (list): First list with proportion of word count.
        sub_list2 (list): Second list with the remaining sentences.
    """
    def get_len(text: str) -> int:
        return len(text.split())

    # Convert the list into a DataFrame
    df = pd.DataFrame(sentences, columns=['sent'])
    df["word_number"] = df['sent'].apply(get_len)
    
    # Remove empty lines
    df = df[df["word_number"] != 0]

    # Calculate the cumulative sum of word counts
    df["cumulative_sum"] = df["word_number"].cumsum()
    
    # Determine the target count for splitting
    total_word_count = df["word_number"].sum()
    target_count = total_word_count * proportion

    # Find the split index based on cumulative sum
    split_index = df[df["cumulative_sum"] >= target_count].index[0] + 1

    # Split the DataFrame
    sub_df1 = df.iloc[:split_index]
    sub_df2 = df.iloc[split_index:]

    
    print(f'The splitted chunk prop: {str(sub_df1["word_number"].sum() / df["word_number"].sum())}')
    return sub_df1['sent'].tolist(), sub_df2['sent'].tolist()



def merge_file(files,loc:Path,step:int,filename:str):

    print(f"Merging files into {str(step)} chunks...")
    # Only iterate while there are enough files for a full group
    for count, i in enumerate(range(0, len(files) - step + 1, step)):  # Adjust the range to exclude the last incomplete group
        group = files[i : i + step]
        merged_text = ""
        
        for file in group:
            # Assuming `file` represents the path to the file and `read_text()` is used to read the file content
            text = '\n'.join(file)
            merged_text += "" + text
        
        # Create the corresponding chunk file for the merged text
        output_file = loc / f"{count:02d}" / filename
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write(merged_text)

    print(f'Finished merging {str(step)} chunks...')


######################################
# assign all the var as the arguments
######################################

proportion = 0.084
lang = "EN"
root_dir: Path = settings.PATH.dataset_root / "STELATranscriptions2"
new_location = root_dir / "by_month" / lang
data_location = root_dir / "txt" / lang / '50h'
dev_location = root_dir / 'dev' / lang 
# create the dir if not existing one
dev_location.mkdir(parents=True, exist_ok=True)
steps: dict = {1:1, 2:2, 3:3, 4:4, 5:6, 9:15,15:25}
filename = "transcription.txt"


###########################################
# split all the datasets into train and dev
###########################################
chunk_list = []
dev_list = []
# Loop through the parent folders in the data_location
for parent_folder in data_location.iterdir():
    if parent_folder.is_dir():  # Check if it is a directory
        # Loop through the .txt files in the current parent folder
        for txt_file in parent_folder.glob("*trancription.txt"):
            with txt_file.open("r", encoding="utf-8") as file:
                sentences = file.read().splitlines()  # Read lines and preserve line structure
                # divide into train and dev
                dev,train = split_by_count(sentences, proportion)
                chunk_list.append(train)  # Add the sentences as a chunk to the list
                dev_list.extend(dev)

# Print the chunk_list (optional, for debugging purposes)
print(len(chunk_list))
print('All the dataset has been loaded')



########################
# write the dev file
########################


# write out the results
with open(dev_location/filename,'w') as f:
     for text in dev_list:
        f.write(text + '\n')

print(f'Finished writing file to {dev_location/filename}')



########################
# merge different chunks
########################

# merge and save files recursively
for month,step in tqdm(steps.items()):
    loc = Path(new_location) / str(month)
    merge_file(chunk_list,loc,step,filename)

print('Finished merging!')







