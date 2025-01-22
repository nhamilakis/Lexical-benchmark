"""replace the given name based on the directory"""
from pathlib import Path
import argparse
import shutil
from lexical_benchmark import settings


def parseargs():
    # Run parameters
    parser = argparse.ArgumentParser(description="Get the array script for generation")
    parser.add_argument("--source1", type=str, default="gen/oberon"
                , help="Source Directory1")
    parser.add_argument("--source2", type=str, default="gen/jz"
                , help="Directory to save path file")
    parser.add_argument("--dest", type=str, default="gen/merged"
                , help="Destination Directory to save path file")
    return parser.parse_args()




def replace_directory_names(root_path: str, name_mapping: dict, target_layers: list) -> None:
    """
    Replace directory names by first looping through target layers, then doing recursive replacement.
    
    Args:
        root_path: Base directory path
        name_mapping: Dictionary of {old_name: new_name}
        target_layers: List of layer names to search in
    """
    root = Path(root_path)
    
    # First loop through target layers
    for layer in target_layers:
        layer_path = root / layer
        if not layer_path.exists():
            print(f"Layer path does not exist: {layer_path}")
            continue
            
        # Get all directories in this layer
        all_dirs = sorted(
            [p for p in layer_path.glob('**/*') if p.is_dir()],
            key=lambda x: len(x.parts),
            reverse=True  # Process deeper directories first
        )
        
        # Replace directory names
        for dir_path in all_dirs:
            dir_name = dir_path.name
            if dir_name in name_mapping:
                new_name = name_mapping[dir_name]
                new_path = dir_path.parent / new_name
                try:
                    dir_path.rename(new_path)
                    print(f"Renamed in {layer}: {dir_path} -> {new_path}")
                except Exception as e:
                    print(f"Error renaming in {layer} - {dir_path}: {e}")



def main():
    # Args parser
    args = parseargs()
    

    # Example usage:
    name_mapping = {
        "ChildRealistic": "Child",
        "old_version": "v2"
    }

    # Specify the layers you want to process
    target_layers = ["models", "outputs", "results"]

    # Example path
    root_path = "/scratch1/projects/lexical-benchmark/v2"

    # Run the replacement
    replace_directory_names(root_path, name_mapping, target_layers)


if __name__ == "__main__":
    main()
