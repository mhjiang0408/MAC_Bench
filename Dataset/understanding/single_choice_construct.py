import os
import sys
sys.path.append(os.getcwd())
import json
import random
from typing import List, Dict, Tuple
import pandas as pd
import argparse
class Construct_Multi_Choice:
    def __init__(self, base_path: str):
        """
        initialize the dataset builder
        Args:
            base_path: Root directory containing Article, Cover, Story and Other_Articles folders
        """
        self.base_path = base_path
        self.article_path = os.path.join(base_path, 'Article')
        self.cover_path = os.path.join(base_path, 'Cover')
        self.story_path = os.path.join(base_path, 'Story')
        self.other_articles_path = os.path.join(base_path, 'Other_Articles')
        
    def get_journals(self) -> List[str]:
        """get all journal names"""
        return [j for j in os.listdir(self.story_path) 
                if os.path.isdir(os.path.join(self.story_path, j))]
    
    def get_issues(self, journal: str) -> List[str]:
        """get all issues of the specified journal
        use cover path, because all journals that can get cover only need to check if there is story and article afterwards"""
        journal_path = os.path.join(self.cover_path, journal)
        return [i.split('.')[0] for i in os.listdir(journal_path) 
                if os.path.isfile(os.path.join(journal_path, i))]
    
    def read_file_content(self, path: str) -> str:
        """read file content"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            return ""
            
    def construct_question(self, journal: str, issue: str) -> Dict:
        """
        construct a multi-choice question for a single issue
        Returns:
            Dict: {
                'id': 'journal_issue',
                'question': 'Which of the following options best describe the cover image?',
                'story': story_content,
                'options': [{'id': 'A', 'text': '...', 'abstract': '...'}, ...],
                'answer': 'A',
                'cover_image': cover_image_path
            }
        """
        
        if not (os.path.exists(os.path.join(self.story_path, journal, f"{issue}.txt")) and os.path.exists(os.path.join(self.other_articles_path, journal, f"{issue}.json"))):
            return None
        story_path = os.path.join(self.story_path, journal, f"{issue}.txt")
        story_content = self.read_file_content(story_path)
        
        
        
        other_path = os.path.join(self.other_articles_path, journal, f"{issue}.json")
        try:
            with open(other_path, 'r') as f:
                other_articles = json.load(f)
        except Exception as e:
            print(f"Error loading other articles for {journal}/{issue}: {e}")
            other_articles = {}
            
        
        options = []
        
                
        option_ids = ['A', 'B', 'C', 'D']
        random.shuffle(option_ids)  # shuffle the option ids
        
        
        options.append({
            'id': option_ids[0],
            'text': story_content,
            'is_correct': True
        })
                
        
        distractors = list(other_articles.items())
        random.shuffle(distractors)
        for i, (url, abstract) in enumerate(distractors[:3]):  # only need 3 distractors
            options.append({
                'id': option_ids[i + 1],  # use the remaining ids
                'text': abstract,
                'is_correct': False
            })
        
        
        options.sort(key=lambda x: x['id'])
        
        
        answer = [opt['id'] for opt in options if opt['is_correct']][0]
        
        
        question = {
            'journal': journal,
            'id': issue,
            'question': 'Which of the following options best describe the cover image?',
            'story': story_content,
            'options': options,
            'answer': answer,
            'cover_image': os.path.join(self.cover_path, journal, f"{issue}.png")
        }
        
        return question
    
    def construct_easy_question(self,journal:str,issue:str,other_stories:list):
        """
        construct a simple single-choice question for a single issue
        """
        
        # if not (os.path.exists(os.path.join(self.story_path, journal, f"{issue}.txt")) and os.path.exists(os.path.join(self.other_articles_path, journal, f"{issue}.json"))):
        #     return None
        if not os.path.exists(os.path.join(self.story_path, journal, f"{issue}.txt")):
            return None
        story_path = os.path.join(self.story_path, journal, f"{issue}.txt")
        story_content = self.read_file_content(story_path)
        
        
        
        # other_path = os.path.join(self.other_articles_path, journal, f"{issue}.json")
        # try:
        #     with open(other_path, 'r') as f:
        #         other_articles = json.load(f)
        # except Exception as e:
        #     print(f"Error loading other articles for {journal}/{issue}: {e}")
        #     other_articles = {}
            
        
        options = []
        
                
        option_ids = ['A', 'B', 'C', 'D']
        random.shuffle(option_ids)  # shuffle the option ids
        
        
        options.append({
            'id': option_ids[0],
            'text': story_content,
            'is_correct': True
        })
                
        
        distractors_path = [s for s in other_stories if s != f"{issue}.txt"]
        random.shuffle(distractors_path)
        
        distractors = []
        for path in distractors_path[:3]:
            story_path = os.path.join(self.story_path, journal, path)
            story_content = self.read_file_content(story_path)
            distractors.append(story_content)
        if not len(distractors) == 3:
            print(f"Error: {journal}/{issue} has less than 3 distractors")
            return None
        for i, cover_story in enumerate(distractors):  # only need 3 distractors
            options.append({
                'id': option_ids[i + 1],  # use the remaining ids
                'text': cover_story,
                'is_correct': False
            })
        
        
        options.sort(key=lambda x: x['id'])
        
        
        answer = [opt['id'] for opt in options if opt['is_correct']][0]
        
        
        question = {
            'journal': journal,
            'id': issue,
            'question': 'Which of the following options best describe the cover image?',
            'story': story_content,
            'options': options,
            'answer': answer,
            'cover_image': os.path.join(self.cover_path, journal, f"{issue}.png")
        }
        
        return question
    
    
    def check_dataset_integrity(self, dataset):
        """
        check the integrity of the dataset
        
        Args:
            dataset: list of questions
            
        Returns:
            valid_dataset: valid dataset after checking
            removed_count: number of removed questions
        """
        valid_dataset = []
        removed_count = 0
        
        for question in dataset:
            is_valid = True
            
            
            if not question['cover_image'] or not os.path.exists(question['cover_image']):
                print(f"the cover image of {question['journal']}/{question['id']} does not exist: {question['cover_image']}")
                is_valid = False
                
            
            if 'option_A' not in question or not question['option_A'] or 'option_B' not in question or not question['option_B'] or 'option_C' not in question or not question['option_C'] or 'option_D' not in question or not question['option_D']:
                print(f"the options of {question['journal']}/{question['id']} does not exist")
                is_valid = False 
            

            if 'answer' not in question or not question['answer']:
                print(f"no correct answer of {question['journal']}/{question['id']}")
                is_valid = False
            
            if is_valid:
                valid_dataset.append(question)
            else:
                removed_count += 1
                
        print(f"data set integrity check completed: total questions {len(dataset)}, valid questions {len(valid_dataset)}, removed questions {removed_count}")
        
        return valid_dataset, removed_count

    def construct_dataset(self, 
                        output_dir: str,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        construct the dataset and save it directly as CSV format
        Args:
            output_dir: output directory path
            train_ratio: training set ratio
            val_ratio: validation set ratio
            seed: random seed
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: training set, validation set and test set
        """
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Construct the dataset
        data_list = []
        journals = self.get_journals()
        

        for journal in journals:
            issues = self.get_issues(journal)
            
            all_stories = os.listdir(os.path.join(self.story_path, journal))
            
            other_stories = [s for s in all_stories] # get the file names

            

            for issue in issues:
                try:
                    # question = self.construct_question(journal, issue)
                    question = self.construct_easy_question(journal, issue, other_stories)
                    if question is None:
                        continue
                    
                    flat_data = {
                        'journal': question['journal'],
                        'id': question['id'],
                        'question': question['question'],
                        'cover_image': question['cover_image'],
                        'answer': question['answer'],
                    }
                    
                    
                    for i, opt in enumerate(question['options']):
                        flat_data[f'option_{opt["id"]}'] = opt["text"]
                    
                    data_list.append(flat_data)
                    
                except Exception as e:
                    print(f"Error processing {journal}/{issue}: {e}")
                    continue
        
        valid_dataset, removed_count = self.check_dataset_integrity(data_list)
        
        df = pd.DataFrame(valid_dataset)
        

        
        random.seed(seed)
        indices = list(range(len(df)))
        random.shuffle(indices)
        
        train_size = int(len(df) * train_ratio)
        val_size = int(len(df) * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_df = df.iloc[train_indices].copy()
        val_df = df.iloc[val_indices].copy()
        test_df = df.iloc[test_indices].copy()
        
        
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        
        
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        for split_name, split_df in splits.items():
            output_path = os.path.join(output_dir, f'{split_name}.csv')
            split_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Saved {split_name} set ({len(split_df)} samples) to {output_path}")
        
        # Save complete dataset
        full_df = pd.concat([train_df, val_df, test_df], axis=0)
        full_output_path = os.path.join(output_dir, 'full_dataset.csv')
        full_df.to_csv(full_output_path, index=False, encoding='utf-8')
        
        
        stats = {
            'total_samples': len(df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': 1 - train_ratio - val_ratio,
            'journals': journals,
            'seed': seed,
            'columns': list(df.columns),
            'removed_count': removed_count
        }
        
        stats_path = os.path.join(output_dir, 'dataset_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        return train_df, val_df, test_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="construct multi-choice dataset")
    parser.add_argument('--output', type=str, default="./Data/Understanding/Nature", required=True, help="output directory")
    parser.add_argument('--base_path', type=str, default="./Nature", required=True, help="base path")
    args = parser.parse_args()
    construct = Construct_Multi_Choice(args.base_path)
    construct.construct_dataset(output_dir=args.output)