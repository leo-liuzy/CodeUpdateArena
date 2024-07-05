"""Utility functions."""

import json
import pickle
import functools

from sqlitedict import SqliteDict

def load_jsonlines(fname: str):
    """Read jsonlines file."""
    with open(fname, 'r') as f:
        return [json.loads(line) for line in f]

def load_json(fname: str):
    """Read json file."""
    with open(fname, 'r') as f:
        return json.load(f)

def dump_json(obj, fname: str, indent: int = None):
    """Dump json file."""
    with open(fname, 'w', encoding='utf-8') as f:
        return json.dump(obj, f, indent=indent)

def load_bin(fname: str):
    """Load binary file."""
    with open(fname, 'rb') as f:
        return pickle.load(f)

def dump_bin(obj, fname: str):
    """Dump binary file."""
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

class SQLiteCache:
    """Cache class using sqlite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = SqliteDict(self.db_path, autocommit=True)

    def cache_func(self, func, hash_func=None):
        """Cache wrapper."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if hash_func is not None:
                key = hash_func(*args, **kwargs)
            else:
                key = f"{func.__name__}:{args}:{kwargs}"
            if key in self.db:
                return self.db[key]
            result = func(*args, **kwargs)
            self.db[key] = result
            return result

        return wrapper

    def close(self):
        """Close the database."""
        self.db.close()