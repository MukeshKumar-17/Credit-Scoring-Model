# main.py - Credit Scoring Project
# Run this to explore your dataset

import os
import sys

# setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data, get_summary, print_summary
from src.utils import make_dir, file_exists
from config.config import (
    RAW_DATA_PATH, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
    MODELS_DIR, REPORTS_DIR
)


def setup_folders():
    """Create project folders if they don't exist"""
    print("\nSetting up folders...")
    for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
        make_dir(folder)
    print("Done!\n")


def explore_data():
    """Load dataset and show basic info"""
    
    if not file_exists(RAW_DATA_PATH):
        print(f"\nDataset not found: {RAW_DATA_PATH}")
        print("Please add your CSV file to data/raw/ folder")
        return None
    
    print("Loading data...")
    df = load_data(RAW_DATA_PATH)
    print("Data loaded!")
    
    # show summary
    summary = get_summary(df)
    print_summary(summary)
    
    # preview first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData types:")
    print(summary["dtypes"])
    
    return df


def main():
    print("\n" + "="*50)
    print("CREDIT SCORING PROJECT")
    print("="*50)
    
    setup_folders()
    df = explore_data()
    
    if df is not None:
        print("\nNext steps:")
        print("1. Check the data summary above")
        print("2. Look for columns that need cleaning")
        print("3. Think about useful features to create")
        print()
    
    return df


if __name__ == "__main__":
    data = main()
