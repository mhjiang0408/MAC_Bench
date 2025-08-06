import cv2
import easyocr
import os
import sys
sys.path.append(os.getcwd())


import os
import argparse
import logging
from tqdm import tqdm
# from utils.clip import CLIPEmbedding

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("image_embedding_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("image_embedding_extractor")

def cover_text_in_image(image_path: str, output_path: str, conf_threshold: float = 0.25) -> None:
    """
    automatically detect all text in the image and cover it with a white rectangle.
    
    Args:
        image_path: input image path to be processed
        output_path: path to save the processed image
        conf_threshold: confidence threshold for text detection (results below this value will be ignored)
    
    Note:
        Please ensure easyocr and OpenCV are installed.
        If easyocr is not installed, you can use pip install easyocr to install it.
    """
    
    image = cv2.imread(image_path)
    if image is None:
        print("Cannot read image:", image_path)
        return False

    
    
    reader = easyocr.Reader(['en'])
    
    
    results = reader.readtext(image)
    
    
    for (bbox, text, prob) in results:
        
        if prob > conf_threshold:
            
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            
            
            cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), -1)
    
    
    cv2.imwrite(output_path, image)
    return True

def process_image_files(base_path, output_base_path=None, model_name="openai/clip-vit-base-patch32"):
    """
    process txt files in all journal folders under base_path, extract text embedding and save
    
    Args:
        base_path: base path, containing journal folders
        output_base_path: output base path, if None, use base_path + "_embeddings"
        model_name: CLIP model name
    """
    
    if output_base_path is None:
        output_base_path = os.path.join(base_path, "OCRed_Cover")
    
    logger.info(f"Start processing text files, base path: {base_path}")
    logger.info(f"Output path: {output_base_path}")
    
    
    # clip_model = OpenClip(model_name="ViT-g-14",pretrained="laion2b_s34b_b88k")
    
    
    
    total_files = 0
    processed_files = 0
    failed_files = 0
    
    story_path = os.path.join(base_path, "Cover")
    if not os.path.exists(story_path):
        logger.error(f"Error: Cover directory does not exist in {base_path}")
        return {}
    
    for journal_name in tqdm(os.listdir(story_path)):
        journal_path = os.path.join(story_path, journal_name)
        if not os.path.isdir(journal_path):
            continue
        logger.info(f"Processing journals: {journal_name}")
        
        journal_output_dir = os.path.join(output_base_path, journal_name)
        os.makedirs(journal_output_dir, exist_ok=True)
        for root, dirs, files in os.walk(journal_path):
            
            txt_files = [f for f in files if f.endswith('.png')]
        
            if not txt_files:
                continue
            
            
            rel_path = os.path.relpath(root, journal_path)
            if rel_path != '.':
                output_dir = os.path.join(journal_output_dir, rel_path)
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = journal_output_dir
            
            logger.info(f"Processing directory: {rel_path} (Found {len(txt_files)} txt files)")
            
            
            for txt_file in txt_files:
                total_files += 1
                
                try:
                    
                    file_path = os.path.join(root, txt_file)
                    

                    
                    
                    output_file = os.path.join(output_dir, txt_file)
                    
                    
                    if os.path.exists(output_file):
                        logger.info(f"File already exists, skipping: {output_file}")
                        processed_files += 1
                        continue
                    
                    
                    success = cover_text_in_image(file_path, output_file,0.25)
                    
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
    parser = argparse.ArgumentParser(description="extract text embedding from txt files")
    parser.add_argument("--base_path", help="base path, containing journal folders")
    parser.add_argument("--output", help="output base path, default is base_path + '_embeddings'")
    
    
    args = parser.parse_args()
    
    try:
        process_image_files(args.base_path, args.output)
    except Exception as e:
        logger.error(f"Script execution error: {str(e)}")



# if __name__ == "__main__":
#     input_img = "./ACS/Cover/ACS Agricultural Science & Technology/2021_1.png"
#     output_img = "./Dataset/test/2021_1.png"
#     cover_text_in_image(input_img, output_img)
