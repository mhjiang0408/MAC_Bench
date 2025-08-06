import pandas as pd
import argparse
import os
import random
from tqdm import tqdm

def merge_and_shuffle_csv(input_files, output_path, seed=None):
    """
    merge multiple CSV files, shuffle all rows, and save to specified path
    
    Args:
        input_files: input CSV file path list
        output_path: output CSV file path
        seed: random seed, for reproducibility
    """
    
    if seed is not None:
        random.seed(seed)
    
    
    all_data = []
    total_rows = 0
    
    
    print("Reading CSV files...")
    for file_path in tqdm(input_files):
        if not os.path.exists(file_path):
            print(f"Warning: file does not exist, skipped: {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            rows = len(df)
            all_data.append(df)
            total_rows += rows
            print(f"Read {file_path}: {rows} rows")
        except Exception as e:
            print(f"Error: error reading file {file_path}: {str(e)}")
    
    if not all_data:
        print("Error: no CSV files were successfully read")
        return False
    
    
    print(f"Merging {len(all_data)} CSV files, total {total_rows} rows...")
    merged_df = pd.concat(all_data, ignore_index=True)
    
    
    print("Shuffling rows...")
    shuffled_df = merged_df.sample(frac=1.0).reset_index(drop=True)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    print(f"Saving to {output_path}...")
    shuffled_df.to_csv(output_path, index=False)
    
    print(f"Done! {len(shuffled_df)} rows data saved to {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="merge multiple CSV files, shuffle all rows, and save to specified path")
    parser.add_argument("--base_path", type=str, help="input CSV file path list")
    parser.add_argument("--output", required=True, help="output CSV file path")
    parser.add_argument("--seed", type=int, help="random seed, for reproducibility")
    
    args = parser.parse_args()
    input_files = [
        f"./Data/Understanding/ACS/{args.base_path}/full_dataset.csv",
        f"./Data/Understanding/Cell/{args.base_path}/full_dataset.csv",
        f"./Data/Understanding/Science/{args.base_path}/full_dataset.csv",
        f"./Data/Understanding/Nature/{args.base_path}/full_dataset.csv",
    ]

    try:
        merge_and_shuffle_csv(input_files, args.output, args.seed)
    except Exception as e:
        print(f"Execution error: {str(e)}")

if __name__ == "__main__":
    main()