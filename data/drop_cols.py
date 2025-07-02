import pandas as pd
import sys

# List of CSV files to process
files = ['test_pro.csv', 'train_pro.csv', 'val_pro.csv']

# Columns to remove
cols_to_remove = ['absolute_path', 'exists']

for file in files:
    df = pd.read_csv(file)
    df = df.drop(columns=cols_to_remove, errors='ignore')
    df.to_csv(file, index=False)