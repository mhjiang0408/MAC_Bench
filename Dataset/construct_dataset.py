import os
import sys
import pandas as pd
import argparse
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_construction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def construct_dataset(base_path, output_path, journal_output_path):
    """
    Construct dataset with Cover as index, organizing all related information into CSV file
    
    Args:
        base_path: Root directory containing Article, Cover, Story and Other_Articles folders
        output_path: Output CSV file path
    """
    # Define subdirectory paths
    cover_path = os.path.join(base_path, 'Cover')
    story_path = os.path.join(base_path, 'Story')
    article_path = os.path.join(base_path, 'Article')
    other_articles_path = os.path.join(base_path, 'Other_Articles')
    
    # Check if directory exists
    if not os.path.exists(cover_path):
        logger.error(f"Cover directory does not exist: {cover_path}")
        return
    
    # Create dataset record lists
    records = []
    empty_records = []
    
    # Get all journal names
    journals = [j for j in os.listdir(cover_path) if os.path.isdir(os.path.join(cover_path, j))]
    logger.info(f"Found {len(journals)} journals")
    
    # Iterate through each journal
    for journal in tqdm(journals, desc="Processing journals"):
        journal_cover_path = os.path.join(cover_path, journal)
        
        # Get all covers under this journal
        covers = [c for c in os.listdir(journal_cover_path) if c.endswith('.png')]
        
        # Iterate through each cover
        for cover in covers:
            # Extract volume and issue numbers
            issue_id = cover.split('.')[0]  # e.g.: "4_10"
            try:
                volume, issue = issue_id.split('_')
            except ValueError:
                logger.warning(f"Cannot parse volume and issue numbers: {cover}")
                continue
            
            # Build various paths
            cover_file_path = os.path.join(cover_path, journal, cover)
            story_file_path = os.path.join(story_path, journal, f"{issue_id}.txt")
            article_file_path = os.path.join(article_path, journal, f"{issue_id}.txt")
            other_articles_file_path = os.path.join(other_articles_path, journal, f"{issue_id}.json")
            
            # Check if files exist
            has_story = os.path.exists(story_file_path)
            has_article = os.path.exists(article_file_path)
            has_other_articles = os.path.exists(other_articles_file_path)
            
            # Create record
            record = {
                'Journal': journal,
                'Volume': volume,
                'Issue': issue,
                'cover image path': f"./Cover/{journal}/{issue_id}.png",
                'cover story path': f"./Story/{journal}/{issue_id}.txt" if has_story else "",
                'cover article path': f"./Article/{journal}/{issue_id}.txt" if has_article else "",
                'articles path': f"./Other_Articles/{journal}/{issue_id}.json" if has_other_articles else ""
            }
            
            # Add to record list
            records.append(record)
            
            # If missing any file, add to empty records list
            if not (has_story):
                empty_records.append(record)
    
    # Create DataFrame and save
    df = pd.DataFrame(records)
    empty_df = pd.DataFrame(empty_records)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete dataset
    df.to_csv(output_path, index=False)
    logger.info(f"Saved complete dataset to {output_path}, total {len(df)} records")
    
    # Save empty records dataset
    empty_output_path = os.path.join(output_dir, 'empty_records.csv')
    empty_df.to_csv(empty_output_path, index=False)
    logger.info(f"Saved empty records dataset to {empty_output_path}, total {len(empty_df)} records")
    
    # Print statistics
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Complete records: {len(df) - len(empty_df)}")
    logger.info(f"Incomplete records: {len(empty_df)}")
    
    # Create DataFrame of complete records (excluding empty records)
    complete_df = df[~df['cover story path'].isna() & (df['cover story path'] != "")]
    
    # Statistics of complete records by journal
    journal_stats = complete_df.groupby('Journal').size().reset_index()
    journal_stats.columns = ['Journal', 'Complete_Records']
    
    # Save journal statistics to CSV
    journal_stats.to_csv(journal_output_path, index=False)
    
    # Print journal statistics for complete records
    logger.info("Complete records by journal:")
    for _, row in journal_stats.iterrows():
        logger.info(f"  {row['Journal']}: {row['Complete_Records']}")
    
    return df, empty_df, journal_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct the dataset based on the cover")
    parser.add_argument('--base_path', type=str, required=True, help="the path of the dataset")
    parser.add_argument('--output', type=str, default="./Data/dataset.csv", help="the path of the output csv file")
    parser.add_argument('--journal_path', type=str, default="./Data/journal_statistics.csv", help="the path of the journal statistics csv file")
    
    args = parser.parse_args()
    
    # Construct the dataset
    df, empty_df, journal_stats = construct_dataset(args.base_path, args.output, args.journal_path)
    print(journal_stats)