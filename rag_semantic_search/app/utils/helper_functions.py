import os
import json
from pathlib import Path

_CURRENT_PATH = Path(__file__).parent.parent
_DATASET_PATH = _CURRENT_PATH / "data" / "data.json"

def read_json():
    """Reads json file.
    """
    with open(_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
    
def preprocess_dataset(data: list):
    ids = [i['id'] for i in data]
    documents = [i['text'] for i in data]

    return ids, documents
    
