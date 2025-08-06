import re
import os
import json
import logging
from bs4 import BeautifulSoup
import requests
from playwright.sync_api import sync_playwright
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("article_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("article_extractor")

def extract_abstract_from_Nature_html(html_content):
    """
    extract abstract content from HTML element
    
    Args:
        html_content: HTML string containing abstract
        
    Returns:
        str: extracted abstract text content
    """
    try:
        
        soup = BeautifulSoup(html_content, 'html.parser')

        abstract_section = soup.find('section', attrs={'data-title': 'Abstract'})
        if abstract_section:
            
            abstract_content = abstract_section.find('div', id=lambda x: x and x.endswith('-content'))
            if not abstract_content:
                logger.warning("In Nature article, Foundabstract section but not Foundcontent div")
                return None
            
            paragraphs = abstract_content.find_all('p')
            abstract_text = ' '.join([p.get_text() for p in paragraphs])
        else:
            abstract_section1 = soup.find('p', class_='article__teaser')
            abstract_section2 = soup.find('meta', attrs={'name': 'description'})
            if not abstract_section1 and not abstract_section2:
                logger.warning("In Nature article, no Found any abstract related elements")
                return None
            
            abstract_text1 = abstract_section1.get_text(strip=True) if abstract_section1 else ""
            abstract_text2 = abstract_section2.get('content') if abstract_section2 else ""

            if len(abstract_text1) < 15:
                abstract_text = abstract_text2
            else:
                abstract_text = abstract_text2 + abstract_text1 

        return abstract_text
    except Exception as e:
        logger.error(f"Error extracting Nature abstract: {str(e)}")
        return None

def extract_abstract_from_Cell_html(html_content):
    """
    extract abstract text from HTML element
    
    Args:
        html_content: HTML string containing abstract
        
    Returns:
        str: extracted abstract text content, keep subscript format
    """
    try:
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        
        abstract_section = soup.find('section', id='author-abstract')
        if not abstract_section:
            logger.warning("In Cell article, no Foundauthor-abstract section")
            return None
        
        
        abstract_divs = abstract_section.find_all('div', role='paragraph')
        if not abstract_divs:
            
            
            for heading in abstract_section.find_all(['h1', 'h2', 'h3']):
                heading.extract()
            abstract_text = abstract_section.get_text(strip=True)
        else:
            
            abstract_text = ' '.join([div.get_text(strip=True) for div in abstract_divs])
        
        
        
        html_content = html_content.replace('<sub>', '§SUB§').replace('</sub>', '§/SUB§')
        html_content = html_content.replace('<sup>', '§SUP§').replace('</sup>', '§/SUP§')
        
        
        soup = BeautifulSoup(html_content, 'html.parser')
        abstract_section = soup.find('section', id='author-abstract')
        if abstract_section:
            abstract_divs = abstract_section.find_all('div', role='paragraph')
            if abstract_divs:
                abstract_text = ' '.join([div.get_text(strip=True) for div in abstract_divs])
        
        
        abstract_text = abstract_text.replace('§SUB§', '_').replace('§/SUB§', '')
        abstract_text = abstract_text.replace('§SUP§', '^').replace('§/SUP§', '')
        
        return abstract_text
    except Exception as e:
        logger.error(f"Error extracting Cell abstract: {str(e)}")
        return None

def extract_abstract_from_Science_html(html_content):
    """
    extract first paragraph text from HTML
    
    Args:
        html_content: HTML string
        
    Returns:
        str: first paragraph text content
    """
    try:
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        
        paragraphs = soup.find_all('div', attrs={'role': 'paragraph'})
        
        
        if paragraphs and len(paragraphs) > 0:
            
            first_paragraph_text = paragraphs[0].get_text()
            
            
            first_paragraph_text = re.sub(r'\s+', ' ', first_paragraph_text).strip()
            
            return first_paragraph_text
        else:
            logger.warning("In Science article, no Foundparagraph element")
            return None
    except Exception as e:
        logger.error(f"Error extracting Science abstract: {str(e)}")
        return None

def extract_abstract_from_ACS_html(html_content):
    """
    extract abstract text from HTML element
    
    Args:
        html_content: HTML string containing abstract
        
    Returns:
        str: extracted abstract text content
    """
    try:
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        
        abstract_div = soup.find('div', class_='article_abstract')
        if not abstract_div:
            logger.warning("In ACS article, no Foundarticle_abstract div")
            return None
        
        
        abstract_content = abstract_div.find('div', class_='article_abstract-content')
        if not abstract_content:
            logger.warning("In ACS article, no Foundarticle_abstract-content div")
            return None
        
        
        conspectus_title = abstract_content.find('h6', class_='article_abstract-sub-title')
        conspectus_text = conspectus_title.get_text().strip() if conspectus_title else ""
        
        
        paragraphs = abstract_content.find_all('p', class_='articleBody_abstractText')
        
        
        abstract_text = ""
        if conspectus_text:
            abstract_text += conspectus_text + "\n\n"
        
        abstract_text += "\n\n".join([p.get_text().strip() for p in paragraphs])
        
        return abstract_text
    except Exception as e:
        logger.error(f"Error extracting ACS abstract: {str(e)}")
        return None

def request_web(year_url):
    """
    use Playwright to request web content
    
    Args:
        year_url: URL to request
        
    Returns:
        str: web content
    """
    try:
        
        proxies_playwright = {
            "server": "http://xxxx",
        }
        headers2 = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'cache-control': 'max-age=0',
            'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            'sec-ch-ua-arch': '"arm"',
            'sec-ch-ua-bitness': '"64"',
            'sec-ch-ua-full-version': '"131.0.6778.265"',
            'sec-ch-ua-full-version-list': '"Google Chrome";v="131.0.6778.265", "Chromium";v="131.0.6778.265", "Not_A Brand";v="24.0.0.0"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-model': '""',
            'sec-ch-ua-platform': '"macOS"',
            'sec-ch-ua-platform-version': '"14.4.0"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }
        
        logger.info(f"Start requesting web: {year_url}")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, proxy=proxies_playwright)
            context = browser.new_context(
                
                viewport={'width': 1920, 'height': 1080},
                accept_downloads=True,
                # User Agent
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                
                
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1'
                },
                
                
                locale='en-US',
                timezone_id='America/New_York',
                geolocation={'latitude': 40.7128, 'longitude': -74.0060},
                permissions=['geolocation']
            )
            page = context.new_page()
            page.set_extra_http_headers(headers2)
            
            
            logger.info(f"Navigate to URL: {year_url}")
            page.goto(year_url, wait_until='networkidle', timeout=60000)
            
            
            try:
                
                try:
                    cookie_button = page.locator('#onetrust-accept-btn-handler')
                    if cookie_button.is_visible(timeout=5000):
                        logger.info("Click Cookie accept button")
                        cookie_button.click()
                except Exception as cookie_e:
                    logger.debug(f"Error processing Cookie prompt or no Cookie prompt: {str(cookie_e)}")
                
                page_content = page.content()
                page.screenshot(path="page2.png", full_page=True)
                logger.info(f"Successfully get page content: {year_url}")
            except Exception as e:
                logger.error(f"Error loading page: {str(e)}")
                page.screenshot(path=f"page_{year_url.replace('://', '_').replace('/', '_')}.png", full_page=True)
                page_content = ""
                
            browser.close()
        return page_content
    except Exception as e:
        logger.error(f"Error requesting web: {str(e)}")
        return ""

def split_text_into_three_paragraphs(text):
    """
    split text into three paragraphs
    
    Args:
        text: text to split
        
    Returns:
        list: list of three paragraphs
    """
    try:
        
        sentence_endings = [m.end() for m in re.finditer(r'\.(?=\s|$)', text)]
        
        if len(sentence_endings) < 3:
            logger.warning(f"Text has less than 3 sentences ({len(sentence_endings)}), return original text")
            return [text]  # if sentence number is less than 3, return original text
        
        
        total_length = len(text)
        ideal_paragraph_length = total_length // 3
        
        
        first_break_idx = min(sentence_endings, key=lambda x: abs(x - ideal_paragraph_length))
        
        
        second_break_idx = min(
            [x for x in sentence_endings if x > first_break_idx], 
            key=lambda x: abs(x - (2 * ideal_paragraph_length))
        )
        
        
        first_paragraph = text[:first_break_idx].strip()
        second_paragraph = text[first_break_idx:second_break_idx].strip()
        third_paragraph = text[second_break_idx:].strip()
        
        return [first_paragraph, second_paragraph, third_paragraph]
    except Exception as e:
        logger.error(f"Error splitting text: {str(e)}")
        return [text]

def read_text_file(file_path):
    """
    read text file and return a list of lines, split by newline
    
    Args:
        file_path: text file path
        
    Returns:
        list: list of lines, split by newline, remove empty lines and whitespace
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            
            lines = [line.strip() for line in file.readlines() if line.strip()]
            return lines
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return []
    
def get_distraction_abstract(url, journal):
    """
    get article abstract based on journal type
    
    Args:
        url: article URL
        journal: journal type
        
    Returns:
        str: extracted abstract text
    """
    try:
        logger.info(f"Start getting {journal} article abstract: {url}")
        proxies = {
            'http': 'http://xxx',
            'https': 'http://xxx',
        }
        
        kv = {
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36'
        }
        
        if journal == 'Nature':
            
            response = requests.get(url, headers=kv, proxies=proxies)
            response.encoding = 'utf-8'
            abstract = extract_abstract_from_Nature_html(response.text)
        elif journal == 'Cell':
            page_content = request_web(url)
            abstract = extract_abstract_from_Cell_html(page_content)
        elif journal == 'Science':
            page_content = request_web(url)
            with open('debug_content.html', 'w') as file:
                file.write(page_content)
            abstract = extract_abstract_from_Science_html(page_content)
        elif journal == 'ACS':
            page_content = request_web(url)
            abstract = extract_abstract_from_ACS_html(page_content)
        else:
            logger.error(f"Unsupported journal type: {journal}")
            return None
        
        if abstract:
            logger.info(f"Successfully get abstract: {url}")
        else:
            logger.warning(f"Failed to get abstract: {url}")
            
        return abstract
    except Exception as e:
        logger.error(f"Error getting abstract {url}: {str(e)}")
        return None

def crawl_all_abstracts(base_path, journal_type='Nature', output_dir=None):
    """
    crawl abstract from all txt files in Article folder under base_path
    
    Args:
        base_path: base path
        journal_type: journal type
        output_dir: output directory, default is Abstract folder under base_path
    
    Returns:
        dict: dictionary containing all crawling results
    """
    try:
        logger.info(f"Start crawling {journal_type} article abstract, base path: {base_path}")
        
        
        if output_dir is None:
            output_dir = os.path.join(base_path, "Description")
        
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        
        article_path = os.path.join(base_path, "Article")
        if not os.path.exists(article_path):
            logger.error(f"Error: Article folder does not exist in {base_path}")
            return {}
        
        
        results = {}
        success_count = 0
        failure_count = 0
        
        
        for journal_name in os.listdir(article_path):
            journal_path = os.path.join(article_path, journal_name)
            
            
            if not os.path.isdir(journal_path):
                continue
            
            logger.info(f"Processing journals: {journal_name}")
            
            
            journal_output_dir = os.path.join(output_dir, journal_name)
            os.makedirs(journal_output_dir, exist_ok=True)
            
            
            for root, dirs, files in os.walk(journal_path):
                
                txt_files = [f for f in files if f.endswith('.txt')]
                
                if not txt_files:
                    continue
                    
                
                rel_dir = os.path.relpath(root, journal_path)
                if rel_dir != '.':
                    current_output_dir = os.path.join(journal_output_dir, rel_dir)
                    os.makedirs(current_output_dir, exist_ok=True)
                else:
                    current_output_dir = journal_output_dir
                
                for txt_file in tqdm(txt_files, desc=f"Processing files in {os.path.relpath(root, article_path)}"):
                    try:
                        file_path = os.path.join(root, txt_file)
                        relative_path = os.path.relpath(file_path, article_path)
                        output_file = os.path.join(current_output_dir, txt_file)
                        if os.path.exists(output_file):
                            logger.info(f"File already exists: {output_file}")
                            continue
                        
                        lines = read_text_file(file_path)
                        
                        if not lines:
                            logger.warning(f"No Found URL: {relative_path}")
                            failure_count += 1
                            continue
                        
                        
                        url = lines[0]  # use the first URL          
                        
                        
                        abstract = get_distraction_abstract(url, journal_type)
                        
                        if abstract:
                            
                            
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(abstract)
                            
                            success_count += 1
                        else:
                            logger.warning(f"Failed to get abstract: {url}")
                            failure_count += 1
                    except Exception as e:
                        logger.error(f"Error processing file {txt_file}: {str(e)}")
                        failure_count += 1
        
        
        all_results_file = os.path.join(output_dir, "all_abstracts.json")
        with open(all_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processing completed, success: {success_count}, failure: {failure_count}, results saved in {output_dir}")
        return results
    except Exception as e:
        logger.error(f"Error crawling abstract: {str(e)}")
        return {}

if __name__ == "__main__":
    try:
        logger.info("Start executing script")
        crawl_all_abstracts("MAC_Bench/Nature", "Nature")
        logger.info("Script executed successfully")
    except Exception as e:
        logger.error(f"Script execution error: {str(e)}")