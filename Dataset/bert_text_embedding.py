import os
import sys
sys.path.append(os.getcwd())


import os
import argparse
import logging
from tqdm import tqdm
from utils.bert import BERTEmbedding

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("text_embedding_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("text_embedding_extractor")

def process_txt_files(base_path, output_base_path=None, model_name="bert-base-uncased"):
    """
    process all txt files in all journal folders under base_path, extract text embedding and save
    
    Args:
        base_path: base path, containing journal folders
        output_base_path: output base path, if None, use base_path + "_embeddings"
        model_name: CLIP model name
    """
    
    if output_base_path is None:
        output_base_path = os.path.join(base_path, "Story_Embeddings_BERT")
    
    logger.info(f"Start processing text files, base path: {base_path}")
    logger.info(f"Output path: {output_base_path}")
    
    
    bert_model = BERTEmbedding(model_name=model_name)
    logger.info(f"CLIP model initialized: {model_name}")
    
    
    total_files = 0
    processed_files = 0
    failed_files = 0
    
    story_path = os.path.join(base_path, "Story")
    if not os.path.exists(story_path):
        logger.error(f"Error: Story folder does not exist in {base_path}")
        return {}
    
    for journal_name in os.listdir(story_path):
        journal_path = os.path.join(story_path, journal_name)
        if not os.path.isdir(journal_path):
            continue
        logger.info(f"Processing journals: {journal_name}")
        
        journal_output_dir = os.path.join(output_base_path, journal_name)
        os.makedirs(journal_output_dir, exist_ok=True)
        for root, dirs, files in os.walk(journal_path):
            
            txt_files = [f for f in files if f.endswith('.txt')]
        
            if not txt_files:
                continue
            
            
            rel_path = os.path.relpath(root, journal_path)
            if rel_path != '.':
                output_dir = os.path.join(journal_output_dir, rel_path)
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = journal_output_dir
            
            logger.info(f"Processing directory: {rel_path} (Found {len(txt_files)} txt files)")
            
            
            for txt_file in tqdm(txt_files, desc=f"Processing {rel_path} files"):
                total_files += 1
                
                try:
                    
                    file_path = os.path.join(root, txt_file)
                    
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        
                        text_content = f.read().strip()
                    
                    if not text_content:
                        logger.warning(f"File is empty: {file_path}")
                        failed_files += 1
                        continue
                    
                    
                    output_file = os.path.join(output_dir, txt_file.replace('.txt', '.pt'))
                    
                    
                    if os.path.exists(output_file):
                        logger.info(f"File already exists, skipping: {output_file}")
                        processed_files += 1
                        continue
                    
                    
                    success = bert_model.get_and_save_text_embedding(text_content, output_file)
                    
                    if success:
                        processed_files += 1
                        logger.debug(f"Successfully processed: {file_path} -> {output_file}")
                    else:
                        failed_files += 1
                        logger.error(f"Processing failed: {file_path}")
                    
                except Exception as e:
                    failed_files += 1
                    logger.error(f"Error processing file {txt_file}: {str(e)}")
    
    
    logger.info(f"Processing completed! Total files: {total_files}, Success: {processed_files}, Failed: {failed_files}")
    return processed_files, failed_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text embedding from txt files")
    parser.add_argument("--base_path", help="Base path containing journal folders")
    parser.add_argument("--output", help="Output base path, default is base_path + '_embeddings'")
    
    
    args = parser.parse_args()
    
    try:
        process_txt_files(args.base_path, args.output,model_name="all-mpnet-base-v2")
    except Exception as e:
        logger.error(f"Script execution error: {str(e)}")