


from pathlib import Path
import pandas as pd
import string

def count_words(text):
    """
    Simple word count after removing punctuation.
    
    Args:
        text (str): Input text to count words
        
    Returns:
        int: Total word count
    """
    # Remove punctuation
    text_no_punct = text.translate(str.maketrans('', '', string.punctuation))
    
    # Split into words and count
    words = text_no_punct.split()
    
    return len(words)

# Set root path
ROOT_PATH = Path("/linkhome/rech/genscp01/uye44va/data/commons/work/lexical-benchmark/datasets/childes/by_size/EN/adult")

# Find all .txt files and count words
count_dict = {}

for txt_file in ROOT_PATH.rglob("*.txt"):
    # Get relative path from root as key
    relative_path = txt_file.relative_to(ROOT_PATH)
    
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        word_count = count_words(text)
        count_dict[str(relative_path)] = word_count
    except Exception as e:
        print(f"Error reading {relative_path}: {e}")

print("Word counts by file:")
print(count_dict)

# Create DataFrame and extract split/chunk information
count_df = pd.DataFrame(list(count_dict.items()), columns=['File', 'Word_Count'])

# Extract split and chunk from file paths
def extract_split_chunk(file_path):
    """Extract split and chunk from file path like '05/03/train.txt' or 'dev/dev.txt'"""
    parts = file_path.split('/')
    
    if parts[0] == 'dev':
        return -1, -1  # Use -1 for dev to sort first
    else:
        # For paths like '05/03/train.txt', convert to integers
        split = int(parts[0])
        chunk = int(parts[1])
        return split, chunk

# Apply extraction and get integer values directly
count_df[['Split', 'Chunk']] = count_df['File'].apply(
    lambda x: pd.Series(extract_split_chunk(x))
)

# Sort by Split and Chunk in ascending order
count_df = count_df.sort_values(['Split', 'Chunk']).reset_index(drop=True)

# Keep only the desired columns
count_df = count_df[['Split', 'Chunk', 'Word_Count']]

print("\nDataFrame with Split and Chunk (sorted):")
print(count_df)

count_df.to_csv(ROOT_PATH/"word_count_stat.csv")
