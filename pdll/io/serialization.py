import json
import io
import base64

from typing import Any
import pickle

# https://docs.python.org/3/library/pickle.html
# __dict__
# __getstate__
# __setstate__

def save(obj: Any, path: str) -> None:
    '''save
    '''
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(path: str) -> Any:
    '''load
    '''
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj