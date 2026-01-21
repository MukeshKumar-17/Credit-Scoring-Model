# data_loader.py - load and inspect datasets

import pandas as pd


def load_data(file_path):
    """Load a CSV file and return as DataFrame"""
    return pd.read_csv(file_path)


def get_summary(df):
    """Get basic info about the dataset"""
    
    # count missing values per column
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        "column": missing.index,
        "missing": missing.values,
        "percent": missing_pct.values
    })
    
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "columns": df.columns.tolist(),
        "missing_info": missing_df,
        "dtypes": df.dtypes
    }


def print_summary(summary):
    """Print dataset summary to console"""
    
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    
    print(f"\nShape: {summary['rows']} rows x {summary['cols']} columns")
    
    print(f"\nColumns:")
    for i, col in enumerate(summary["columns"], 1):
        print(f"  {i}. {col}")
    
    print(f"\nMissing Values:")
    missing_df = summary["missing_info"]
    has_missing = missing_df[missing_df["missing"] > 0]
    
    if len(has_missing) == 0:
        print("  No missing values!")
    else:
        for _, row in has_missing.iterrows():
            print(f"  - {row['column']}: {row['missing']} ({row['percent']}%)")
    
    print("="*50 + "\n")
