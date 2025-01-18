from datetime import datetime
from pathlib import Path
import shutil

def merge_folders_with_conflict_resolution(src_folder1, src_folder2, dst_folder, strategy='ask'):
    """
    Merge folders with conflict resolution.
    
    Args:
        src_folder1 (str): Path to first source folder
        src_folder2 (str): Path to second source folder
        dst_folder (str): Path to destination folder
        strategy (str): Conflict resolution strategy ('ask', 'newest', 'skip', 'overwrite')
    """
    Path(dst_folder).mkdir(parents=True, exist_ok=True)
    
    def get_file_info(path):
        return {
            'mtime': datetime.fromtimestamp(path.stat().st_mtime),
            'size': path.stat().st_size
        }
    
    def resolve_conflict(file1, file2, rel_path):
        if strategy == 'newest':
            info1 = get_file_info(file1)
            info2 = get_file_info(file2)
            return file2 if info2['mtime'] > info1['mtime'] else file1
        elif strategy == 'skip':
            return None
        elif strategy == 'overwrite':
            return file2
        elif strategy == 'ask':
            while True:
                choice = input(f"\nConflict for {rel_path}\n"
                             f"1: Keep version from folder1 ({file1})\n"
                             f"2: Keep version from folder2 ({file2})\n"
                             f"s: Skip this file\n"
                             "Choose [1/2/s]: ")
                if choice == '1':
                    return file1
                elif choice == '2':
                    return file2
                elif choice.lower() == 's':
                    return None
        return None

    # Process both folders
    for item in Path(src_folder1).glob('**/*'):
        if item.is_file():
            rel_path = item.relative_to(src_folder1)
            dst_path = Path(dst_folder) / rel_path
            src2_path = Path(src_folder2) / rel_path
            
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            if src2_path.exists():
                # Conflict found
                resolved_path = resolve_conflict(item, src2_path, rel_path)
                if resolved_path:
                    shutil.copy2(resolved_path, dst_path)
                    print(f"Resolved conflict for: {rel_path}")
            else:
                # No conflict
                shutil.copy2(item, dst_path)
                print(f"Copied from folder1: {rel_path}")

    # Copy remaining files from folder2
    for item in Path(src_folder2).glob('**/*'):
        if item.is_file():
            rel_path = item.relative_to(src_folder2)
            dst_path = Path(dst_folder) / rel_path
            if not dst_path.exists():
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst_path)
                print(f"Copied from folder2: {rel_path}")

# Example usage
if __name__ == "__main__":
    folder1 = "/scratch1/projects/lexical-benchmark/v2/gen/oberon"
    folder2 = "/scratch1/projects/lexical-benchmark/v2/gen/jz"
    destination = "/scratch1/projects/lexical-benchmark/v2/gen/merged"
    
    # Choose strategy: 'ask', 'newest', 'skip', or 'overwrite'
    merge_folders_with_conflict_resolution(folder1, folder2, destination, strategy='ask')