import pandas as pd
import argparse

def find_matching_records(small_csv: str, large_csv: str, output_csv: str):
    """
    find records in large CSV file that match records in small CSV file
    
    Args:
        small_csv: small CSV file path
        large_csv: large CSV file path
        output_csv: output CSV file path
    """
    try:
        print(f"Reading small CSV file: {small_csv}")
        small_df = pd.read_csv(small_csv)
        print(f"Small CSV file has {len(small_df)} records")
        
        print(f"Reading large CSV file: {large_csv}")
        large_df = pd.read_csv(large_csv)
        print(f"Large CSV file has {len(large_df)} records")
        
        small_df['id'] = small_df['id'].astype(str)
        large_df['id'] = large_df['id'].astype(str)
        
        
        
        matched_records = pd.merge(
            large_df,
            small_df[['journal', 'id']],
            on=['journal', 'id'],
            how='inner'
        )
        
        
        if matched_records.empty:
            print("No matching records found!")
            return
        
        
        matched_records.to_csv(output_csv, index=False)
        
        print(f"\nProcessing completed!")
        print(f"Found {len(matched_records)} matching records")
        print(f"Results saved to: {output_csv}")
        
        
        merged = pd.merge(
            small_df[['journal', 'id']],
            large_df[['journal', 'id']],
            on=['journal', 'id'],
            how='left',
            indicator=True
        )
        unmatched = merged[merged['_merge'] == 'left_only']
        
        if not unmatched.empty:
            print(f"\nWarning: {len(unmatched)} records in large CSV file not found in small CSV file:")
            for _, row in unmatched.iterrows():
                print(f"journal: {row['journal']}, id: {row['id']}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise
if __name__ == "__main__":
    mac_2025 = "MAC_Bench/image2text_info.csv"
    total_2026 = "MAC_Bench/image2text_info.csv"
    output = "MAC_Bench/image2text_info.csv"
    find_matching_records(mac_2025, total_2026, output)