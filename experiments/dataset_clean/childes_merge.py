
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
from tqdm import tqdm
from lexical_benchmark import settings
from pathlib import Path

lang = "EN"
root_dir: Path = settings.PATH.dataset_root / "STELATranscriptions2"
new_location = root_dir / "by_month" / lang
data_location = root_dir / "txt" / lang / '50h'
steps: dict = {1:1, 2:2, 3:3, 4:4, 5:6, 9:15,15:25}



########################
# load all the datasets
########################
chunk_list = []
# Loop through the parent folders in the data_location
for parent_folder in data_location.iterdir():
    if parent_folder.is_dir():  # Check if it is a directory
        # Loop through the .txt files in the current parent folder
        for txt_file in parent_folder.glob(f"*.txt"):
            with txt_file.open("r", encoding="utf-8") as file:
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
    loc = Path(new_location) / str(month)
    merge_file(chunk_list,loc,step)

print('Finished merging!')