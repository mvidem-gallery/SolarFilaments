import os
import json
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def data_exists(dir_path) -> bool:
    root_path = get_project_root()

    if not os.path.exists(root_path / dir_path) or not os.listdir(root_path / dir_path):
        return False

    return True


def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data
