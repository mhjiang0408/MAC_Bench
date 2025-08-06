import os
import sys
sys.path.append(os.getcwd())
import json
import random
from typing import List, Dict, Tuple
import pandas as pd
import argparse

from tqdm import tqdm
class Construct_Multi_Choice:
    def __init__(self, base_path: str, num_options: int = 4, embedding_type: str = "given",given_info:str= 'image2text'):
        """
        Initialize dataset builder
        Args:
            base_path: Root directory containing Article, Cover, Story and Other_Articles folders
            num_options: number of options, default is 4
        """
        self.base_path = base_path
        self.given_info = given_info
        self.embedding_type = embedding_type
        self.article_path = os.path.join(base_path, 'Article')
        self.cover_path = os.path.join(base_path, 'Cover')
        self.story_path = os.path.join(base_path, 'Story')
        self.story_embedding_path = os.path.join(base_path, 'Story_Embeddings_CLIP')
        self.cover_embedding_path = os.path.join(base_path, 'Cover_Embeddings_CLIP')
        self.story_bert_embedding_path = os.path.join(base_path, 'Story_Embeddings_multiBv1')
        if embedding_type == "option" and given_info == "image2text":
            self.similarity_paths = ['allv2','multiBv1','sbert','ave']
        else:
            self.similarity_paths = ['siglip','doubao','qwen','ave']

        self.other_articles_path = os.path.join(base_path, 'Other_Articles')
        if 'Nature' in base_path:
            self.tag = 'Nature'
        elif 'Cell' in base_path:
            self.tag = 'Cell'
        elif 'Science' in base_path:
            self.tag = 'Science'
        elif 'ACS' in base_path:
            self.tag = 'ACS'
        self.num_options = num_options
        self.option_ids = [chr(65 + i) for i in range(num_options)]  # 65 is the ASCII code for 'A'
        
        
    def get_journals(self) -> List[str]:
        """get all journal names"""
        return [j for j in os.listdir(self.story_path) 
                if os.path.isdir(os.path.join(self.story_path, j))]
    
    def get_jourals_cover_path(self) -> List[str]:
        """get all journal names"""
        return [j for j in os.listdir(self.cover_path) 
                if os.path.isdir(os.path.join(self.cover_path, j))]

    def get_issues(self, journal: str) -> List[str]:
        """get all issues of the specified journal
        use cover path, because all that can get cover only needs to check if there is story and article afterwards"""
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
        use other articles of the same issue as distractors
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
        random.shuffle(option_ids)
        
        
        options.append({
            'id': option_ids[0],
            'text': story_content,
            'is_correct': True
        })
                
        
        distractors = list(other_articles.items())
        random.shuffle(distractors)
        for i, (url, abstract) in enumerate(distractors[:3]):  # only need 3 distractors
            options.append({
                'id': option_ids[i + 1],  # use remaining IDs
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
    
    def construct_text2image_given_domain_question(self, journal:str, issue:str, other_stories:list):
        """
        given info vs groundtruth
        """
        if not os.path.exists(os.path.join(self.story_path, journal, f"{issue}.txt")):
            
            return None
        story_path = os.path.join(self.story_path, journal, f"{issue}.txt")
        image_path = os.path.join(self.cover_path, journal, f"{issue}.png")
        story_content = self.read_file_content(story_path)

        
        options = []
        
        option_ids = self.option_ids.copy()
        random.shuffle(option_ids)
        
        
        options.append({
            'id': option_ids[0],
            'text': image_path,
            'is_correct': True,
            'path': image_path,
            'embedding_name': "groundtruth",
            'embedding_id':f'{issue}.txt'
        })

        
        similarity_paths = []
        sim_datas = {}
        for path in self.similarity_paths:
            sim_name = self.given_info + "_" + self.embedding_type + "_" + path
            similarity_paths.append(os.path.join(self.base_path, sim_name, journal, f"{issue}.json"))
        for id,sim_path in enumerate(similarity_paths):
            with open(sim_path, 'r') as f:
                sim = json.load(f)
            sim_datas[self.similarity_paths[id]] = sim

        
        distractors_path = {}
        for embedding_name, sim_data in sim_datas.items():
            if embedding_name == "ave":
                continue
            for data in sim_data:
                if data[0] == f"{issue}.txt":
                    continue
                
                if data[0] not in distractors_path.values():
                    distractors_path[embedding_name] = data[0]
                break
        if len(distractors_path.values()) < (self.num_options - 1):
            
            ave_data = sim_datas["ave"]
            for index,data in enumerate(ave_data):
                if data[0] == f"{issue}.txt":
                    continue
                if data[0] not in distractors_path.values():
                    distractors_path[f"ave_{index}"] = data[0]
                if len(distractors_path.values()) == (self.num_options - 1):
                    break
        
        if len(distractors_path.values()) < (self.num_options - 1):
            print(f"Error: {journal}/{issue} has less than {self.num_options-1} distractors")
            return None
        
        
        distractors = {}
        distractors_full_path = []
        for embedding_name, path in distractors_path.items():
            distractors[embedding_name] = [os.path.join(self.cover_path, journal, path.replace(".txt", ".png")), path]
            distractors_full_path.append(os.path.join(self.cover_path, journal, path.replace(".txt", ".png")))
        
        count = 0
        
        
        for embedding_name, data in distractors.items():
            options.append({
                'id': option_ids[count+1],
                'text': data[0],
                'is_correct': False,
                'path': distractors_full_path[count],
                'embedding_name': embedding_name,
                'embedding_id': data[1]
            })
            count += 1
        
        
        options.sort(key=lambda x: x['id'])
        
        
        answer = [opt['id'] for opt in options if opt['is_correct']][0]
        
        
        question = {
            'journal': journal,
            'id': issue,
            'question': 'Which of the following options best describe the cover story?',
            'story': story_content,
            'options': options,
            'answer': answer,
            'cover_image':  story_content
        }
        
        return question
    
    def construct_text2image_domain_question(self, journal:str, issue:str, other_stories:list):
        """
        given info vs groundtruth
        """
        if not os.path.exists(os.path.join(self.story_path, journal, f"{issue}.txt")):
            
            return None
        story_path = os.path.join(self.story_path, journal, f"{issue}.txt")
        image_path = os.path.join(self.cover_path, journal, f"{issue}.png")
        story_content = self.read_file_content(story_path)
        
        options = []
        
        option_ids = self.option_ids.copy()
        random.shuffle(option_ids)
        
        
        options.append({
            'id': option_ids[0],
            'text': image_path,
            'is_correct': True,
            'path': image_path,
            'embedding_name': "groundtruth",
            'embedding_id':f'{issue}.txt'
        })

        
        # distractors_path = [s for s in other_stories if s != f"{issue}.txt"]
        similarity_paths = []
        sim_datas = {}
        for path in self.similarity_paths:
            sim_name = self.given_info + "_" + self.embedding_type + "_" + path
            similarity_paths.append(os.path.join(self.base_path, sim_name, journal, f"{issue}.json"))
        for id,sim_path in enumerate(similarity_paths):
            with open(sim_path, 'r') as f:
                sim = json.load(f)
            sim_datas[self.similarity_paths[id]] = sim

        
        distractors_path = {}
        for embedding_name, sim_data in sim_datas.items():
            if embedding_name == "ave":
                continue
            for data in sim_data:
                if data[0] == f"{issue}.txt":
                    continue
                
                if data[0] not in distractors_path.values():
                    distractors_path[embedding_name] = data[0]
                break
        if len(distractors_path.values()) < (self.num_options - 1):
            
            ave_data = sim_datas["ave"]
            for index,data in enumerate(ave_data):
                if data[0] == f"{issue}.txt":
                    continue
                if data[0] not in distractors_path.values():
                    distractors_path[f"ave_{index}"] = data[0]
                if len(distractors_path.values()) == (self.num_options - 1):
                    break
        
        if len(distractors_path.values()) < (self.num_options - 1):
            print(f"Error: {journal}/{issue} has less than {self.num_options-1} distractors")
            return None
        
        
        distractors = {}
        distractors_full_path = []
        for embedding_name, path in distractors_path.items():
            distractors[embedding_name] = [os.path.join(self.cover_path, journal, path.replace(".txt", ".png")), path]
            distractors_full_path.append(os.path.join(self.cover_path, journal, path.replace(".txt", ".png")))
        
        count = 0
        
        
        for embedding_name, data in distractors.items():
            options.append({
                'id': option_ids[count+1],
                'text': data[0],
                'is_correct': False,
                'path': distractors_full_path[count],
                'embedding_name': embedding_name,
                'embedding_id': data[1]
            })
            count += 1
        
        
        options.sort(key=lambda x: x['id'])
        
        
        answer = [opt['id'] for opt in options if opt['is_correct']][0]
        
        
        question = {
            'journal': journal,
            'id': issue,
            'question': 'Which of the following options best describe the cover story?',
            'story': story_content,
            'options': options,
            'answer': answer,
            'cover_image': story_content
        }
        
        return question

    def construct_domain_question(self, journal:str, issue:str, other_stories:list):
        """
        clip embedding vs groundtruth
        """
        if not os.path.exists(os.path.join(self.story_path, journal, f"{issue}.txt")):
            
            return None
        story_path = os.path.join(self.story_path, journal, f"{issue}.txt")
        story_content = self.read_file_content(story_path)
        
        options = []
        
        option_ids = self.option_ids.copy()
        random.shuffle(option_ids)
        
        
        options.append({
            'id': option_ids[0],
            'text': story_content,
            'is_correct': True,
            'path': story_path,
            'embedding_name': "groundtruth",
            'embedding_id':f'{issue}.txt'
        })
        

        
        # distractors_path = [s for s in other_stories if s != f"{issue}.txt"]
        similarity_paths = []
        sim_datas = {}
        for path in self.similarity_paths:
            sim_name = self.given_info + "_" + self.embedding_type + "_" + path
            similarity_paths.append(os.path.join(self.base_path, sim_name, journal, f"{issue}.json"))
        for id,sim_path in enumerate(similarity_paths):
            with open(sim_path, 'r') as f:
                sim = json.load(f)
            sim_datas[self.similarity_paths[id]] = sim

        
        distractors_path = {}
        for embedding_name, sim_data in sim_datas.items():
            if embedding_name == "ave":
                continue
            for data in sim_data:
                if data[0] == f"{issue}.txt":
                    continue
                
                if data[0] not in distractors_path.values():
                    distractors_path[embedding_name] = data[0]
                break
        if len(distractors_path.values()) < (self.num_options - 1):
            
            ave_data = sim_datas["doubao"]
            for index,data in enumerate(ave_data):
                if data[0] == f"{issue}.txt":
                    continue
                if data[0] not in distractors_path.values():
                    distractors_path[f"doubao_{index}"] = data[0]
                if len(distractors_path.values()) == (self.num_options - 1):
                    break

        if len(distractors_path.values()) < (self.num_options - 1):
            print(f"Error: {journal}/{issue} has less than {self.num_options-1} distractors")
            print(distractors_path)
            return None
        
        
        distractors = {}
        distractors_full_path = []
        for embedding_name, path in distractors_path.items():
            distractors[embedding_name] = [self.read_file_content(os.path.join(self.story_path, journal, path)), path]
            distractors_full_path.append(os.path.join(self.story_path, journal, path))
        count = 0
        
        
        for embedding_name, data in distractors.items():
            options.append({
                'id': option_ids[count+1],
                'text': data[0],
                'is_correct': False,
                'path': distractors_full_path[count],
                'embedding_name': embedding_name,
                'embedding_id': data[1]
            })
            count += 1
        
        
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
    
    def construct_bert_domain_question(self, journal:str, issue:str, other_stories:list):
        """
        given info vs groundtruth
        """
        if not os.path.exists(os.path.join(self.story_path, journal, f"{issue}.txt")):
            
            return None
        story_path = os.path.join(self.story_path, journal, f"{issue}.txt")
        story_content = self.read_file_content(story_path)
        
        options = []
        
        option_ids = self.option_ids.copy()
        random.shuffle(option_ids)
        
        
        options.append({
            'id': option_ids[0],
            'text': story_content,
            'is_correct': True,
            'path': story_path,
            'embedding_name': "groundtruth",
            'embedding_id':f'{issue}.txt'
        })

        
        # distractors_path = [s for s in other_stories if s != f"{issue}.txt"]
        similarity_paths = []
        sim_datas = {}
        for path in self.similarity_paths:
            sim_name = self.given_info + "_" + self.embedding_type + "_" + path
            similarity_paths.append(os.path.join(self.base_path, sim_name, journal, f"{issue}.json"))
        for id,sim_path in enumerate(similarity_paths):
            with open(sim_path, 'r') as f:
                sim = json.load(f)
            sim_datas[self.similarity_paths[id]] = sim

        
        distractors_path = {}
        for embedding_name, sim_data in sim_datas.items():
            if embedding_name == "ave":
                continue
            for data in sim_data:
                if data[0] == f"{issue}.txt":
                    continue
                
                if data[0] not in distractors_path.values():
                    distractors_path[embedding_name] = data[0]
                break
        if len(distractors_path.values()) < (self.num_options - 1):
            
            ave_data = sim_datas["ave"]
            for index,data in enumerate(ave_data):
                if data[0] == f"{issue}.txt":
                    continue
                if data[0] not in distractors_path.values():
                    distractors_path[f"ave_{index}"] = data[0]
                if len(distractors_path.values()) == (self.num_options - 1):
                    break
        
        if len(distractors_path.values()) < (self.num_options - 1):
            print(f"Error: {journal}/{issue} has less than {self.num_options-1} distractors")
            return None
        
        
        distractors = {}
        distractors_full_path = []
        for embedding_name, path in distractors_path.items():
            distractors[embedding_name] = [self.read_file_content(os.path.join(self.story_path, journal, path)), path]
            distractors_full_path.append(os.path.join(self.story_path, journal, path))
        
        count = 0
        
        
        for embedding_name, data in distractors.items():
            options.append({
                'id': option_ids[count+1],
                'text': data[0],
                'is_correct': False,
                'path': distractors_full_path[count],
                'embedding_name': embedding_name,
                'embedding_id': data[1]
            })
            count += 1
        
        
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
        """check dataset integrity"""
        valid_dataset = []
        removed_count = 0
        if self.given_info == 'image2text':
            for question in dataset:
                is_valid = True
                
                if not question['cover_image'] or not os.path.exists(question['cover_image']):
                    print(f"the cover image of {question['journal']}/{question['id']} does not exist: {question['cover_image']}")
                    is_valid = False
                
                
                for option_id in self.option_ids:
                    if f'option_{option_id}' not in question or not question[f'option_{option_id}']:
                        print(f"option_{option_id} of {question['journal']}/{question['id']} does not exist")
                        is_valid = False
                        break
                
                
                if 'answer' not in question or not question['answer']:
                    print(f"no correct answer of {question['journal']}/{question['id']}")
                    is_valid = False
                
                if is_valid:
                    valid_dataset.append(question)
                else:
                    removed_count += 1
        elif self.given_info == 'text2image':
            for question in dataset:
                is_valid = True
                if not question['cover_image']:
                    print(f"the cover image of {question['journal']}/{question['id']} does not exist: {question['cover_image']}")
                    is_valid = False
                for option_id in self.option_ids:
                    if f'option_{option_id}' not in question or not question[f'option_{option_id}']:
                        print(f"option_{option_id} of {question['journal']}/{question['id']} does not exist")
                        is_valid = False
                        break
                    if not os.path.exists(question[f'option_{option_id}']):
                        print(f"the option_{option_id} of {question['journal']}/{question['id']} does not exist: {question[f'option_{option_id}']}")
                        is_valid = False
                        break
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
        Construct the dataset and save it directly as CSV format
        Args:
            output_dir: output directory path
            train_ratio: training set ratio
            val_ratio: validation set ratio
            seed: random seed
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: training set, validation set and test set DataFrame
        """
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Construct the dataset
        data_list = []

        journals = self.get_journals()
        

        for journal in tqdm(journals):
            issues = self.get_issues(journal)
            
            all_stories = os.listdir(os.path.join(self.story_path, journal))
            
            other_stories = [s for s in all_stories] # 这里得到的是文件名


            for issue in issues:
                try:
                    # question = self.construct_question(journal, issue)
                    # question = self.construct_easy_question(journal, issue, other_stories)
                    if self.given_info == "text2image" and self.embedding_type == "given":
                        question = self.construct_text2image_given_domain_question(journal, issue, other_stories)
                    elif self.given_info == "text2image" and self.embedding_type == "option":
                        question = self.construct_text2image_domain_question(journal, issue, other_stories)
                    elif self.given_info == "image2text" and self.embedding_type == "given":
                        question = self.construct_domain_question(journal, issue, other_stories)
                    elif self.given_info == "image2text" and self.embedding_type == "option":
                        question = self.construct_bert_domain_question(journal, issue, other_stories)
                    # question = self.construct_text2image_domain_question(journal, issue, other_stories)

                    # question = self.construct_domain_question(journal, issue, other_stories)
                    # question = self.construct_bert_domain_question(journal, issue, other_stories)
                    # question, question2 = self.construct_double_question(journal, issue, other_stories)

                    # if question is None or question2 is None:
                    #     continue
                    if question is None:
                        continue
                    
                    flat_data = {
                        'journal': question['journal'],
                        'id': question['id'],
                        'question': question['question'],
                        'cover_image': question['cover_image'],
                        'answer': question['answer']
                    }
                    # flat_data2 = {
                    #     'journal': question2['journal'],
                    #     'id': question2['id'],
                    #     'question': question2['question'],
                    #     'cover_image': question2['cover_image'],
                    #     'answer': question2['answer'],
                    # }
                    
                    for i, opt in enumerate(question['options']):
                        flat_data[f'option_{opt["id"]}'] = opt["text"]
                        flat_data[f'option_{opt["id"]}_path'] = opt["path"]
                        flat_data[f'option_{opt["id"]}_embedding_name'] = opt["embedding_name"]
                        flat_data[f'option_{opt["id"]}_embedding_id'] = opt["embedding_id"]
                    # for i, opt in enumerate(question2['options']):
                    #     flat_data2[f'option_{opt["id"]}'] = opt["text"]
                    
                    data_list.append(flat_data)
                    # data_list.append(flat_data2)
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
    parser.add_argument('--num_options', type=int, default=8, help="number of options (default: 8)")
    parser.add_argument('--embedding_type', type=str, default="given", help="embedding type (default: clip)")
    parser.add_argument('--given_info', type=str, default="image2text", help="given info (default: image2text)")
    args = parser.parse_args()
    
    construct = Construct_Multi_Choice(args.base_path, num_options=args.num_options, embedding_type=args.embedding_type, given_info=args.given_info)
    construct.construct_dataset(output_dir=args.output)