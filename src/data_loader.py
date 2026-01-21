# data_loader.py - load dataset

import pandas as pd


def load_data(path):
    """Load german credit numeric dataset with column names"""
    # create column names: Feature_1 to Feature_24, plus Target
    columns = [f"Feature_{i}" for i in range(1, 25)] + ["Target"]
    
    # load data (space-separated, no header)
    df = pd.read_csv(path, sep=r'\s+', header=None, names=columns)
    return df
