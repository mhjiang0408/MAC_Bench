import pandas as pd
import os
import argparse

def extract_volume_number(question_id):
    """
    extract volume number from question_id (the number before the underscore)
    for example: "227_2" -> 227
    """
    try:
        return int(question_id.split('_')[0])
    except:
        return 0

from PIL import Image
import numpy as np

def check_white_patches(image_path: str, white_threshold: int = 240, patch_ratio: float = 0.9) -> bool:
    """
    check if the image has large white patches
    
    Args:
        image_path: image path
        white_threshold: pixel threshold for white pixels (0-255), default 240
        patch_ratio: white patch ratio threshold, default 0.75 (75%)
        
    Returns:
        bool: True if there are large white patches, False if the image is normal
    """
    try:
        
        img = Image.open(image_path)
        
        
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        
        img_array = np.array(img)
        
        
        pixel_means = np.mean(img_array, axis=2)
        
        
        white_pixels = np.sum(pixel_means > white_threshold)
        total_pixels = pixel_means.size
        
        
        white_ratio = white_pixels / total_pixels
        
        
        if white_ratio > patch_ratio:
            
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return True  # if error, consider the image有问题

def check_all_images_in_row(row) -> bool:
    """
    check if all images in a row have problems
    
    Args:
        row: a row of DataFrame
        
    Returns:
        bool: True if all images have problems, False if at least one image is normal
    """
    
    # cover_image = row['cover_image']
    # if 'png' in cover_image:
    #     if not check_white_patches(cover_image):
    #         return False
        
    
    # for opt in ['A', 'B', 'C', 'D']:
    #     option_path = row[f'option_{opt}_path']
    #     if not '.png' in option_path:
    #         continue
    #     if not check_white_patches(option_path):
    #         return False
    
    ground_truth = row['answer']
    
    
    option_path = row[f'option_{ground_truth}_path']
    if '.png' in option_path:
        if check_white_patches(option_path):
            
            return True
    
    
    
    return False

def filter_latest_data(input_path, output_path, min_questions=10, random_seed=42):
    """
    filter the latest issue for each journal, if the number of questions in the latest issue is less than min_questions, select the latest two issues
    also check and delete records with all images having large white patches
    """
    
    print(f"Reading dataset: {input_path}")
    df = pd.read_csv(input_path)
    
    
    df['volume'] = df['id'].apply(extract_volume_number)
    
    
    journals_info = []
    
    
    all_filtered = []
    for journal in df['journal'].unique():
        print(f"\nProcessing journals: {journal}")
        journal_df = df[df['journal'] == journal].copy()
        
        
        journal_df = journal_df.sort_values('volume', ascending=True)
        
        
        valid_volumes = []
        for volume in journal_df['volume'].unique():
            volume_df = journal_df[journal_df['volume'] == volume]
            
            
            valid_rows = []
            for _, row in volume_df.iterrows():
                if not check_all_images_in_row(row):
                    valid_rows.append(row)
            
            if len(valid_rows) >= min_questions:
                valid_volumes.append(volume)
                selected_df = pd.DataFrame(valid_rows)
                break  # stop when the first issue with enough questions is found
        
        if not valid_volumes:
            print(f"Warning: {journal} has no issue with enough questions")
            continue
            
        
        journals_info.append({
            "journal": journal,
            "total_questions": len(journal_df),
            "selected_questions": len(selected_df),
            "volume_min": int(selected_df['volume'].min()),
            "volume_max": int(selected_df['volume'].max()),
            "selected_volumes": ','.join(map(str, valid_volumes)),
            "selected_volumes_count": len(valid_volumes)
        })
        
        print(f"  Total data: {len(journal_df)}")
        print(f"  Selected data: {len(selected_df)}")
        print(f"  Selected volumes: {valid_volumes}")
        
        all_filtered.append(selected_df)
    
    
    filtered_df = pd.concat(all_filtered, ignore_index=True)
    
    
    filtered_df = filtered_df.drop('volume', axis=1)
    
    
    filtered_df = filtered_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    
    filtered_df.to_csv(output_path, index=False)
    info_path = os.path.splitext(output_path)[0] + '_info.csv'
    pd.DataFrame(journals_info).to_csv(info_path, index=False)
    
    print(f"\nFiltering completed!")
    print(f"Original dataset size: {len(df)}")
    print(f"Filtered dataset size: {len(filtered_df)}")
    print(f"Results saved to: {output_path}")
    print(f"Journal information saved to: {info_path}")
    
    return filtered_df, pd.DataFrame(journals_info)

def main():
    parser = argparse.ArgumentParser(description="filter the latest issue for each journal, if the number of questions in the latest issue is less than min_questions, select the latest two issues")
    parser.add_argument('--input', type=str, required=True, help="input CSV file path")
    parser.add_argument('--output', type=str, help="output CSV file path")
    parser.add_argument('--min_questions', type=int, default=12, help="minimum number of questions, if the latest issue has less than this number, select the latest two issues, default 12")
    parser.add_argument('--seed', type=int, default=42, help="random seed, default 42")
    
    args = parser.parse_args()
    
    
    if args.output is None:
        input_name = os.path.splitext(args.input)[0]
        args.output = f"{input_name}_filtered.csv"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    
    filter_latest_data(args.input, args.output, args.min_questions, args.seed)
if __name__ == "__main__":
    main()