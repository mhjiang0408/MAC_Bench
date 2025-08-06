import os
from PIL import Image
from collections import Counter
import pandas as pd
from tqdm import tqdm
import argparse

def analyze_image_sizes(base_path):
    """
    分析目录下所有图片的尺寸
    
    Args:
        base_path: 图片目录的根路径
    
    Returns:
        DataFrame: 包含尺寸统计信息的DataFrame
    """
    
    sizes = []
    errors = []
    total_files = 0
    story_path = os.path.join(base_path, "Cover")
    if not os.path.exists(story_path):
        print(f"Error: Story directory does not exist in {base_path}")
        return {}
    
    for journal_name in tqdm(os.listdir(story_path)):
        journal_path = os.path.join(story_path, journal_name)
        if not os.path.isdir(journal_path):
            continue
        print(f"Processing journals: {journal_name}")
        
        # journal_output_dir = os.path.join(output_base_path, journal_name)
        # os.makedirs(journal_output_dir, exist_ok=True)
        for root, dirs, files in os.walk(journal_path):
            
            txt_files = [f for f in files if f.endswith('.png')]
        
            if not txt_files:
                continue
            
            
            # rel_path = os.path.relpath(root, journal_path)
            # if rel_path != '.':
            #     output_dir = os.path.join(journal_output_dir, rel_path)
            #     os.makedirs(output_dir, exist_ok=True)
            # else:
            #     output_dir = journal_output_dir
            
            
            
            
            for txt_file in txt_files:
                total_files += 1
                
                try:
                    
                    file_path = os.path.join(root, txt_file)
                    with Image.open(file_path) as img:
                        sizes.append({
                            'path': file_path,
                            'width': img.size[0],
                            'height': img.size[1],
                            'size': f"{img.size[0]}x{img.size[1]}"
                        })
                except Exception as e:
                    errors.append({
                        'path': file_path,
                        'error': str(e)
                    })

    
    df = pd.DataFrame(sizes)
    
    
    size_counts = df['size'].value_counts().reset_index()
    size_counts.columns = ['Resolution', 'Count']
    
    
    stats = {
        'Total Images': len(sizes),
        'Unique Sizes': len(size_counts),
        'Failed Images': len(errors),
        'Min Width': df['width'].min(),
        'Max Width': df['width'].max(),
        'Min Height': df['height'].min(),
        'Max Height': df['height'].max(),
        'Mean Width': df['width'].mean(),
        'Mean Height': df['height'].mean()
    }
    
    return df, size_counts, stats, errors

def save_results(df, size_counts, stats, errors, output_dir):
    """
    save analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    
    df.to_csv(os.path.join(output_dir, 'all_images.csv'), index=False)
    
    
    size_counts.to_csv(os.path.join(output_dir, 'size_statistics.csv'), index=False)
    
    
    pd.DataFrame([stats]).to_csv(os.path.join(output_dir, 'basic_statistics.csv'), index=False)
    
    
    if errors:
        pd.DataFrame(errors).to_csv(os.path.join(output_dir, 'errors.csv'), index=False)

def main():
    parser = argparse.ArgumentParser(description="analyze image size statistics")
    parser.add_argument('--input', type=str, required=True, help="image directory path")
    parser.add_argument('--output', type=str, default="./image_stats", help="output directory path")
    
    args = parser.parse_args()
    
    print(f"Start analyzing directory: {args.input}")
    df, size_counts, stats, errors = analyze_image_sizes(args.input)
    
    
    print("\n=== Basic statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Most common image sizes ===")
    print(size_counts.head().to_string(index=False))
    
    
    save_results(df, size_counts, stats, errors, args.output)
    print(f"\nResults saved to: {args.output}")
    
    if errors:
        print(f"\nWarning: {len(errors)} images failed, see errors.csv")

if __name__ == "__main__":
    main()
