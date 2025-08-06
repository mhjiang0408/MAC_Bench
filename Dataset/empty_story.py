import os
import csv
import argparse
from tqdm import tqdm
import pandas as pd

def find_missing_embeddings(base_path, output_csv):
    """
    find issue files that exist in Story directory but not in Story_Embedding_ViTH directory
    
    Args:
        base_path: base path, containing Story and Story_Embedding_ViTH directories
        output_csv: output CSV file path
    """
    
    story_dir = os.path.join(base_path, "Story")
    embedding_dir = os.path.join(base_path, "Story_Embeddings_ViTH14")
    
    # Check if directory exists
    if not os.path.exists(story_dir):
        raise ValueError(f"Story directory does not exist: {story_dir}")
    if not os.path.exists(embedding_dir):
        raise ValueError(f"Story_Embedding_ViTH directory does not exist: {embedding_dir}")
    
    
    missing_files = []
    
    
    print("Scanning Story directory...")
    for journal in tqdm(os.listdir(story_dir)):
        journal_path = os.path.join(story_dir, journal)
        
        
        if not os.path.isdir(journal_path):
            continue
        
        
        embedding_journal_path = os.path.join(embedding_dir, journal)
        if not os.path.exists(embedding_journal_path):
            
            for story_file in os.listdir(journal_path):
                if story_file.endswith('.txt'):
                    story_file_path = os.path.join(journal_path, story_file)
                    missing_files.append({
                        'journal': journal,
                        'issue': story_file,
                        'story_path': story_file_path,
                        'reason': 'missing_journal_dir'
                    })
            continue
        
        
        for story_file in os.listdir(journal_path):
            if not story_file.endswith('.txt'):
                continue
                
            story_file_path = os.path.join(journal_path, story_file)
            
            
            embedding_file = story_file.replace('.txt', '.pt')
            embedding_file_path = os.path.join(embedding_journal_path, embedding_file)
            
            
            if not os.path.exists(embedding_file_path):
                missing_files.append({
                    'journal': journal,
                    'issue': story_file,
                    'story_path': story_file_path,
                    'reason': 'missing_embedding_file'
                })
    
    
    print(f"Found {len(missing_files)} missing embedding files")
    
    if missing_files:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['journal', 'issue', 'story_path', 'reason']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for file_info in missing_files:
                writer.writerow(file_info)
        
        print(f"Results saved to: {output_csv}")
    else:
        print("No missing embedding files found")

def find_missing_cover(base_path, output_csv):
    """
    find issue files that exist in Story directory but not in Story_Cover directory
    """
    
    story_dir = os.path.join(base_path, "Story")
    embedding_dir = os.path.join(base_path, "Cover")
    
    # Check if directory exists
    if not os.path.exists(story_dir):
        raise ValueError(f"Story directory does not exist: {story_dir}")
    if not os.path.exists(embedding_dir):
        raise ValueError(f"Story_Cover directory does not exist: {embedding_dir}")
    
    
    missing_files = []
    
    
    print("Scanning Story directory...")
    for journal in tqdm(os.listdir(story_dir)):
        journal_path = os.path.join(story_dir, journal)
        
        
        if not os.path.isdir(journal_path):
            continue
        
        
        embedding_journal_path = os.path.join(embedding_dir, journal)
        if not os.path.exists(embedding_journal_path):
            
            for story_file in os.listdir(journal_path):
                if story_file.endswith('.txt'):
                    story_file_path = os.path.join(journal_path, story_file)
                    missing_files.append({
                        'journal': journal,
                        'issue': story_file,
                        'story_path': story_file_path,
                        'reason': 'missing_journal_dir'
                    })
            continue
        
        
        for story_file in os.listdir(journal_path):
            if not story_file.endswith('.txt'):
                continue
                
            story_file_path = os.path.join(journal_path, story_file)
            
            
            embedding_file = story_file.replace('.txt', '.png')
            embedding_file_path = os.path.join(embedding_journal_path, embedding_file)
            
            
            if not os.path.exists(embedding_file_path):
                missing_files.append({
                    'journal': journal,
                    'issue': story_file,
                    'story_path': story_file_path,
                    'reason': 'missing_embedding_file'
                })
    
    
    print(f"Found {len(missing_files)} missing embedding files")
    
    if missing_files:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['journal', 'issue', 'story_path', 'reason']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for file_info in missing_files:
                writer.writerow(file_info)
        
        print(f"Results saved to: {output_csv}")
    else:
        print("No missing embedding files found")

def delete_files_from_csv(csv_path, dry_run=True):
    """
    read story_path column from CSV file and delete these files
    
    Args:
        csv_path: CSV file path
        dry_run: if True, only print the files to be deleted without actually deleting
        
    Returns:
        tuple: (number of successfully deleted files, number of failed files)
    """
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file does not exist: {csv_path}")
        return 0, 0
    
    
    try:
        
        df = pd.read_csv(csv_path)
        
        
        if 'story_path' not in df.columns:
            print("Error: CSV file must contain 'story_path' column")
            return 0, 0
        
        
        file_paths = [path for path in df['story_path'] if path and os.path.exists(path)]
        
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return 0, 0
    
    print(f"Read {len(file_paths)} valid file paths from CSV file")
    
    
    if dry_run:
        print("=== Dry run mode - the following files will be deleted ===")
        for path in file_paths[:10]:  # 只显示前10个
            print(f"Will delete: {path}")
        if len(file_paths) > 10:
            print(f"... and {len(file_paths) - 10} other files")
        print(f"Total {len(file_paths)} files")
        print("To actually delete files, set dry_run parameter to False")
        return 0, 0
    
    
    success_count = 0
    failure_count = 0
    
    print("Deleting files...")
    for file_path in tqdm(file_paths):
        try:
            os.remove(file_path)
            success_count += 1
        except Exception as e:
            print(f"Error deleting file {file_path}: {str(e)}")
            failure_count += 1
    
    print(f"Deletion completed! Success: {success_count}, Failure: {failure_count}")
    return success_count, failure_count

def main():
    parser = argparse.ArgumentParser(description="find issue files that exist in Story directory but not in Story_Embedding_ViTH directory")
    parser.add_argument("--base_path", help="base path, containing Story and Story_Embedding_ViTH directories")
    parser.add_argument("--output_path", default="missing_embeddings.csv", help="output CSV file path")
    parser.add_argument("--execute", action="store_true", help="whether to execute the mode")
    
    args = parser.parse_args()
    
    try:
        if not args.execute:
            # find_missing_embeddings(args.base_path, args.output_path)
            find_missing_cover(args.base_path, args.output_path)
        else:
            delete_files_from_csv(args.base_path, dry_run=False)
    except Exception as e:
        print(f"Execution error: {str(e)}")

if __name__ == "__main__":
    main()
    