"""
this python script is used to fix the source of the dataset which cannot get some attributes while spidering.
"""
import pandas as pd
import os
from datetime import datetime
import argparse
import json
def extract_empty_records(input_path: str, output_path: str):
    """
    extract records with empty values from CSV file
    
    Args:
        input_path: input CSV file path
        output_path: output CSV file path
    """
    
    df = pd.read_csv(input_path)
    
    
    empty_records = df[df.isna().any(axis=1)]
    
    
    if not empty_records.empty:

        
        empty_records.to_csv(output_path, index=False)
        return empty_records
    else:
        print("No empty records found!")
        return None


if __name__ == "__main__":
    
    
    input_path = 'MAC_Bench/image2text_info.csv'
    output_path = 'MAC_Bench/image2text_info_fixed.csv'
    
    extract_empty_records(input_path, output_path)