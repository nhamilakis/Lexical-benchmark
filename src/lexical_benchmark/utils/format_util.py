from pathlib import Path

def str_to_bool(value):
    """
    Convert string representation of boolean to actual boolean value.
    Handles various common string formats like 'true', 'false', '1', '0', etc.
    
    Args:
        value (str): String to convert

    Returns:
        bool: Corresponding boolean value
    """
    if isinstance(value, bool):
        return value
        
    value = str(value).lower().strip()
    
    true_values = {'true', '1', 'yes', 'y', 'on'}
    false_values = {'false', '0', 'no', 'n', 'off'}
    
    if value in true_values:
        return True
    elif value in false_values:
        return False
    else:
        raise ValueError(f"Cannot convert '{value}' to boolean")




class DirectoryFilter:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def get_subdirs(self) -> list:
        """Get immediate subdirectories from a path."""
        return [d for d in self.root_dir.iterdir() if d.is_dir()]

    def get_sorted_subdirs(self, directory_list: list, max_count: int) -> list:
        """Get the required number of subdirectories, sorted numerically."""
        try:
            sorted_dirs = sorted(directory_list, key=lambda x: int(x.name))
            return sorted_dirs[:max_count]
        except ValueError:
            print(f"Warning: Some subdirectory names in {self.root_dir} are not numeric. Using default sorting.")
            return directory_list[:max_count]

    def filter_subdirs_by_count(self, max_count: int) -> list:
        """Get filtered subdirectories based on count."""
        subdirs = self.get_subdirs()
        if max_count > 0:
            subdirs = self.get_sorted_subdirs(subdirs, max_count)
        return subdirs

    def filter_subdirs_by_name(self, target_names: list) -> list:
        """Filter subdirectories based on target names."""
        subdirs = self.get_subdirs()
        if len(target_names) > 0:
            return [d for d in subdirs if d.name in target_names]
        return subdirs


