
"""

"""
import os
from tqdm import tqdm
from lexical_benchmark import settings
from pathlib import Path

lang = "EN"
root_dir: Path = settings.PATH.dataset_root / "ChildRealistic"
new_location = root_dir / "by_month" / lang
data_location = root_dir / "txt" / lang / '50h'
steps: dict = {6:6}


########################
# load all the datasets
########################
chunk_list = []
# Loop through the parent folders in the data_location
for parent_folder in data_location.iterdir():
    if parent_folder.is_dir():  # Check if it is a directory
        # Loop through the .txt files in the current parent folder
        with open(parent_folder/"transcription.txt", encoding="utf-8") as file:
            sentences = file.read().splitlines()  # Read lines and preserve line structure
            chunk_list.append(sentences)  # Add the sentences as a chunk to the list

# Print the chunk_list (optional, for debugging purposes)
print(len(chunk_list))
print('All the dataset has been loaded')


########################
# merge different chunks
########################


def merge_file(files,loc:Path,step:int):

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
        output_file = loc / f"{count:02d}" / "transcription.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write(merged_text)

    print(f'Finished merging {str(step)} chunks...')


# merge and save files recursively

for month,step in tqdm(steps.items()):
    loc = Path(new_location) / f"{month:02d}"
    merge_file(chunk_list,loc,step)

print('Finished merging!')