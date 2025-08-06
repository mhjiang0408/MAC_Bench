import pandas as pd
import json
import os
from typing import Dict, List, Tuple
import os
import sys
import re
sys.path.append(os.getcwd())
import argparse
from utils.parse_jsonString import parse_json_string

def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    load original dataset
    
    Args:
        dataset_path: dataset file path
    
    Returns:
        DataFrame: dataset containing journal and id
    """
    try:
        df = pd.read_csv(dataset_path)
        required_columns = ['journal', 'id']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"dataset missing required columns: {required_columns}")
        return df
    except Exception as e:
        print(f"error loading dataset: {e}")
        raise

def load_experiment_results(results_path: str) -> pd.DataFrame:
    """
    load experiment results
    
    Args:
        results_path: experiment results file path
    
    Returns:
        DataFrame: dataset containing experiment results
    """
    try:
        df = pd.read_csv(results_path)
        required_columns = ['journal', 'question_id', 'options']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"experiment results missing required columns: {required_columns}")
        return df
    except Exception as e:
        print(f"error loading experiment results: {e}")
        raise

def extract_description(text):
    
    text = text.replace('\n','')
    pattern = r'"description":"(.*?)"(?=,"options")'
    match = re.search(pattern, text)
    if "I'm sorry" in text:
        return 'delete'
    if match:
        return match.group(1)
    return "No description found"

def process_datasets(dataset_path: str, results_path: str, output_path: str):
    """
    process dataset and experiment results, extract options description, and delete records with description 'delete'
    
    Args:
        dataset_path: original dataset path
        results_path: experiment results path
        output_path: output file path
    """
    try:
        dataset = load_dataset(dataset_path)
        print(f"successfully loaded dataset, {len(dataset)} records")
        
        results = load_experiment_results(results_path)
        print(f"successfully loaded experiment results, {len(results)} records")
        
        dataset['id'] = dataset['id'].astype(str)
        results['question_id'] = results['question_id'].astype(str)
        
        merged = pd.merge(
            dataset,
            results[['journal', 'question_id', 'options']],
            left_on=['journal', 'id'],
            right_on=['journal', 'question_id'],
            how='left'
        )
        
        
        data_list = []
        for _, row in merged.iterrows():
            if pd.isna(row['options']):
                description = "No matching experiment results found"
            else:
                description = extract_description(row['options'])
                
            
            if description != 'delete':
                data_list.append({
                    'journal': row['journal'],
                    'id': row['id'],
                    'question': row['question'],
                    'description': description
                })
            # data_list.append({
            #         'journal': row['journal'],
            #         'id': row['id'],
            #         'question': row['question'],
            #         'description': description
            #     })
        
        
        output_df = pd.DataFrame(data_list)
        
        
        output_df.to_csv(output_path, index=False)
        print(f"results saved to: {output_path}")
        
        # Print statistics
        total_original = len(merged)
        total_after = len(output_df)
        print(f"original records: {total_original}")
        print(f"processed records: {total_after}")
        print(f"deleted records: {total_original - total_after}")
        
    except Exception as e:
        print(f"error processing data: {e}")
        raise


if __name__ == "__main__":
    dataset_path = "MAC_Bench/image2text_info.csv"
    results_path = "experiment/results/understanding/CoVR-gpt-4o_4options_20250318_210605/results.csv"
    output_path = "MAC_Bench/image2text_option/deleted.csv"
    process_datasets(dataset_path, results_path, output_path)
