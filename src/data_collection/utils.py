import json
import os

def save_to_file(data, filename):
    """Saves a given piece of data to a file in json format.

    Args:
        data (object): The data to be saved to the file
        filename (str): The filename to save the data to
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def create_dir(path):
    """Creates a directory if it does not already exist.

    Args:
        path (str): The path to the directory to be created
    """
    if not os.path.exists(path):
        os.makedirs(path)