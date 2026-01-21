# config.py - project settings

import os

# paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# dataset - change this to your file name
DATASET_FILENAME = "german_credit.csv"
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, DATASET_FILENAME)

# model settings
RANDOM_SEED = 42
TEST_SIZE = 0.2
