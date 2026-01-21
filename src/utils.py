# utils.py - helper functions

import os


def make_dir(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created: {path}")


def file_exists(path):
    """Check if file exists"""
    return os.path.exists(path)
