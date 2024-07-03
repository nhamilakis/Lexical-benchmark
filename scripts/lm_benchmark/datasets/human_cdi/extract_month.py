"""update month info from the CHILDES transcripts"""
import re
from pandas import pd


ROOT_path = '/scratch1/projects/lexical-benchmark/datasets/sources/CHILDES/transcript/'

def get_path(file_path:str):
    # Substring to search for
    search_str = 'cleaned_transcript/'
    # Find the position of the substring
    pos = file_path.find(search_str)
    # Extract the substring starting from the found position
    if pos != -1:
        extracted_path = file_path[pos+19:-4]
    new_path = ROOT_path + extracted_path + '.cha'
    return new_path


def find_month(lst, substring):
    for element in lst:
        if substring in element:
            return element
    return None  # Return None if no element is found

def convert_month(string:str)->int:
    # Define the regular expression pattern to match the age component (e.g., '2;06.')
    pattern = r'(\d+);(\d+)\.'
    # Find the match
    match = re.search(pattern, string)
    if match:
        # Extract the year and month parts
        years = int(match.group(1))
        months = int(match.group(2))
        # Convert the age to total months
        total_months = years * 12 + months
    return total_months

def extract_month(file_path:str):
    try:
        trans = get_path(file_path)
        with open(trans,'r') as f:
            content = f.readlines()
            try:
                target_str = find_month(content, 'CHI|')
                print(f"Target sent: {target_str}")
                new_month = convert_month(target_str)
                print(f'Convert int the month {str(new_month)}')
            except:
                print('something wrong with extracting ')
                new_month = 'Placeholder'
        return new_month
    except:
        return "Placeholder"

