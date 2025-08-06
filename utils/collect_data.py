import os
import pandas as pd
from pathlib import Path

def create_dataset(base_dir: str) -> pd.DataFrame:
    """
    create a dataset by collecting cover, story, and article data
    
    Args:
        base_dir: base directory path
    
    Returns:
        pd.DataFrame: DataFrame containing all data samples information
    """
    data = []
    
    
    cover_base = os.path.join(base_dir, 'Cover')
    story_base = os.path.join(base_dir, 'Story')
    article_base = os.path.join(base_dir, 'Article')
    
    
    if os.path.exists(cover_base):
        journals = [d for d in os.listdir(cover_base) if os.path.isdir(os.path.join(cover_base, d))]
        
        for journal in journals:
            print(f"Processing journal: {journal}")
            
            cover_journal_dir = os.path.join(cover_base, journal)
            story_journal_dir = os.path.join(story_base, journal)
            article_journal_dir = os.path.join(article_base, journal)
            
            
            cover_files = Path(cover_journal_dir).rglob('*.*')
            
            for cover in cover_files:
                
                cover_path = os.path.relpath(cover, base_dir)
                
                
                relative_path = os.path.relpath(cover, cover_journal_dir)
                relative_path_no_ext = os.path.splitext(relative_path)[0]  # 移除原始扩展名
                potential_story = os.path.join(story_journal_dir, relative_path_no_ext + '.txt')
                potential_article = os.path.join(article_journal_dir, relative_path_no_ext + '.txt')
                
                
                story_path = os.path.relpath(potential_story, base_dir) if os.path.exists(potential_story) else None
                article_path = os.path.relpath(potential_article, base_dir) if os.path.exists(potential_article) else None
                
                data.append({
                    'journal': journal,
                    'cover_path': cover_path,
                    'story_path': story_path,
                    'article_path': article_path
                })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    
    base_dir = "./Cell"
    
    
    dataset = create_dataset(base_dir)
    
    
    output_file = "dataset_cell.csv"
    dataset.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    print(f"Total samples: {len(dataset)}")