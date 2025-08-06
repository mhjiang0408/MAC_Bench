import os
import sys
sys.path.append(os.getcwd())
import base64
from experiment.method.DAD import CoVRr0,CoVRr3
from utils.config_loader import ConfigLoader
from utils.parse_jsonString import parse_probabilities
from utils.llm_call import CallLLM
from utils.test_api import test_model_api
from typing import List, Dict, Tuple
import openai
import json
import csv, os, re
import random
import argparse
import pandas as pd
from tqdm import tqdm
import wandb
from datetime import datetime
import os
import logging
import time
from functools import wraps
import multiprocessing  # Add multiprocessing support
import math  # Add math functions for group calculations
import threading
import queue

# logger = logging.getLogger('journal_processor')

# def setup_logger(log_file: str = "./debug/journal_processing.log") -> None:
#     """

#     """

#     os.makedirs(os.path.dirname(log_file), exist_ok=True)
    

#     logger.setLevel(logging.INFO)
    

#     if logger.handlers:
#         logger.handlers.clear()
    

#     file_handler = logging.FileHandler(log_file, encoding='utf-8')
#     file_handler.setLevel(logging.INFO)
    

#     formatter = logging.Formatter('%(asctime)s - %(message)s')
#     file_handler.setFormatter(formatter)
    

#     logger.addHandler(file_handler)



class MultiChoiceEvaluation:
    def __init__(self, model:str = "Qwen/Qwen2.5-7B-Instruct", 
                 api_base:str = "https://xxx/v1", 
                 api_key:str = "sk-xxx", 
                 prompt_template:dict[str, str] = None,
                 num_options:int = 4,
                 type:str = "image2text"):  # Add option count parameter
        self.model = model
        if 'ocr' in self.model:
            self.ocr = True
        else:
            self.ocr = False
        if 'few_shot' in self.model:
            self.few_shot = True
        else:
            self.few_shot = False
        if 'descripted' in self.model:
            self.descripted = True
        else:
            self.descripted = False
        self.model = self.model.replace('_few_shot','')
        self.model = self.model.replace('_ocr','')
        self.model = self.model.replace('_descripted','')
        self.api_base = api_base
        self.api_key = api_key
        self.majority = False
        self.prompt_template = prompt_template or loader.load_config('Config/prompt_template/template.json')
        if self.model =='CoVR':
            if self.ocr:
                self.llm = CoVRr3(ocr=True,mllm_api_base=self.api_base, mllm_api_key=self.api_key)
            else:
                self.llm = CoVRr3(mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        elif self.model == 'CoVR-qwen-max':
            if self.ocr:
                self.llm = CoVRr3(ocr=True,mllm_model='qwen-vl-max',mllm_api_base=self.api_base, mllm_api_key=self.api_key)
            else:
                self.llm = CoVRr3(mllm_model='qwen-vl-max',mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        elif self.model == 'CoVR-step-1v-8k':
            if self.ocr:
                self.llm = CoVRr3(ocr=True,mllm_model='step-1v-8k',mllm_api_base=self.api_base, mllm_api_key=self.api_key)
            else:
                self.llm = CoVRr3(mllm_model='step-1v-8k',mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        elif self.model == 'CoVR-step-1o-turbo-vision':
            if self.ocr:
                self.llm = CoVRr3(ocr=True,mllm_model='step-1o-turbo-vision',mllm_api_base=self.api_base, mllm_api_key=self.api_key)
            else:
                self.llm = CoVRr3(mllm_model='step-1o-turbo-vision',mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        elif self.model == 'CoVR-gemini':
            if self.ocr:
                self.llm = CoVRr3(ocr=True,mllm_model='gemini-1.5-pro-latest', mllm_api_base=self.api_base, mllm_api_key=self.api_key)
            else:
                self.llm = CoVRr3(mllm_model='gemini-1.5-pro-latest', mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        elif self.model == 'CoVR-ernie-4.5-8k':
            if self.ocr:
                self.llm = CoVRr3(ocr=True,mllm_model='ernie-4.5-8k-preview', low_detail=True,mllm_api_base=self.api_base, mllm_api_key=self.api_key)
            else:
                self.llm = CoVRr3(mllm_model='ernie-4.5-8k-preview', low_detail=True,mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        elif self.model == 'CoVR-gpt-4o':
            if self.ocr:
                self.llm = CoVRr3(ocr=True,mllm_model='gpt-4o', mllm_api_base=self.api_base, mllm_api_key=self.api_key)
            else:
                self.llm = CoVRr3(mllm_model='gpt-4o', mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        elif self.model == 'CoVR-r1':
            if self.ocr:
                self.llm = CoVRr3(ocr=True,reasoning_model='deepseek-r1-250120', reasoning_api_base=self.api_base, reasoning_api_key=self.api_key)
            else:
                self.llm = CoVRr3(reasoning_model='deepseek-r1-250120', reasoning_api_base=self.api_base, reasoning_api_key=self.api_key)
        elif self.model == 'CoVR-o3-mini':
            if self.ocr:
                self.llm = CoVRr3(ocr=True,reasoning_model='o3-mini',reasoning_api_base=self.api_base, reasoning_api_key=self.api_key)
            else:
                self.llm = CoVRr3(reasoning_model='o3-mini',reasoning_api_base=self.api_base, reasoning_api_key=self.api_key)
        else:
            print(f"model {self.model}")
            self.llm = CallLLM(model=self.model, api_base=self.api_base, api_key=self.api_key)
        self.type = type
        self.num_options = num_options
        
        self.option_ids = [chr(65 + i) for i in range(num_options)]  # 65 is ASCII code for 'A'

    def split_dataset(self,data: pd.DataFrame, num_splits: int = 5):
        """
        Split dataset evenly into specified number of parts
        
        Args:
            data: Original dataset
            num_splits: Number of splits
        
        Returns:
            list: List containing split datasets
        """
        
        split_size = len(data) // num_splits
        remainder = len(data) % num_splits
        
        splits = []
        start = 0
        for i in range(num_splits):
            
            end = start + split_size + (1 if i < remainder else 0)
            splits.append(data.iloc[start:end].copy())
            start = end
        
        return splits
    def multi_choice_message(self, question:str, options:str, image_path:str):
        """
        Multiple choice prompt
        """
        if self.few_shot:
            system_prompt = self.prompt_template['system_prompt']
            user_prompt = self.prompt_template['user_prompt']
            few_shot_prompt = self.prompt_template['few_shot_prompt']
            few_shot_image = self.prompt_template['few_shot_image']
        else:
            system_prompt = self.prompt_template['system_prompt']
            user_prompt = self.prompt_template['user_prompt']
        user_content = user_prompt.format(
            question=question, 
            options=options
        )
        if self.ocr:
            image_path = image_path.replace('Cover','OCRed_Cover')
        try:
            with open(image_path, 'rb') as image_file:
                
                base64_data = base64.b64encode(image_file.read())
                
                image_base64 = base64_data.decode('utf-8')
                if self.few_shot:
                    few_shot_base64 = base64.b64encode(open(few_shot_image, 'rb').read())
                    few_shot_base64 = few_shot_base64.decode('utf-8')

        except Exception as e:
            print(f"Error: {e}")
            return None
        if self.few_shot:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": user_content
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{few_shot_base64}",
                            "detail": "low"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "low"
                        }
                    }
                ]}
            ]
        else:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": user_content
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "low"
                        }
                    }
                ]}
            ]
    
    def multi_choice_message_text2image(self, question:str, story:str, options:list, low_detail:bool = False):
        """
        Multiple choice prompt for text2image
        """
        if self.few_shot:
            system_prompt = self.prompt_template['system_prompt']
            user_prompt = self.prompt_template['user_prompt']
            few_shot_story = self.prompt_template['few_shot_story']
            few_shot_optionA = self.prompt_template['few_shot_optionA']
            few_shot_optionB = self.prompt_template['few_shot_optionB']
            few_shot_optionC = self.prompt_template['few_shot_optionC']
            few_shot_optionD = self.prompt_template['few_shot_optionD']
        else:
            system_prompt = self.prompt_template['system_prompt']
            user_prompt = self.prompt_template['user_prompt']
        user_content = user_prompt.format(
            question=question, 
            story=story
        )
        base64_list = []
        for option in options:
            try:
                if self.ocr:
                    option = option.replace('Cover','OCRed_Cover')
                with open(option, 'rb') as image_file:
                    
                    base64_data = base64.b64encode(image_file.read())
                    
                    image_base64 = base64_data.decode('utf-8')
                    base64_list.append(image_base64)
            except Exception as e:
                print(f"Error: {e}")
                return None
        if self.few_shot:
            few_shot_imageA = base64.b64encode(open(few_shot_optionA, 'rb').read()).decode('utf-8')
            few_shot_imageB = base64.b64encode(open(few_shot_optionB, 'rb').read()).decode('utf-8')
            few_shot_imageC = base64.b64encode(open(few_shot_optionC, 'rb').read()).decode('utf-8')
            few_shot_imageD = base64.b64encode(open(few_shot_optionD, 'rb').read()).decode('utf-8')
        if low_detail:
            detail = "low"
        else:
            detail = "low"
        if self.few_shot:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": user_content
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{few_shot_imageA}",
                            "detail": detail
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{few_shot_imageB}",
                            "detail": detail
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{few_shot_imageC}",
                            "detail": detail
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{few_shot_imageD}",
                            "detail": detail
                        }
                    },
                    {
                        "type": "text",
                        "text": few_shot_story
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_list[0]}",
                            "detail": detail
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_list[1]}",
                            "detail": detail
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_list[2]}",
                            "detail": detail
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_list[3]}",
                            "detail": detail
                        }
                    }
                ]}
            ]
        else:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": user_content
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_list[0]}",
                            "detail": detail
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_list[1]}",
                            "detail": detail
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_list[2]}",
                            "detail": detail
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_list[3]}",
                            "detail": detail
                        }
                    }
                ]}
            ]

    
    @staticmethod
    def prepare_dataset(data_path: str, scaling_factor: float = 1.0, seed: int = 42):
        """
        Prepare dataset, shuffle using random seed
        
        Args:
            data_path: dataset path
            scaling_factor: sampling ratio
            seed: random seed for reproducibility
            
        Returns:
            shuffled dataset
        """
        data = pd.read_csv(data_path)
        
        if scaling_factor < 1.0:
            
            return data.sample(frac=scaling_factor, random_state=seed)
        else:
            
            return data.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    
    def writer_process(self, csv_path, fieldnames, result_queue, stop_event):
        """
        Dedicated writer process that reads results from queue and writes to CSV file
        
        Args:
            csv_path: CSV file path
            fieldnames: CSV file field names
            result_queue: result queue
            stop_event: stop event to notify writer process to end
        """
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            
            while not stop_event.is_set() or not result_queue.empty():
                try:
                    
                    result = result_queue.get(timeout=1)
                    
                    
                    writer.writerow(result['record'])
                    csvfile.flush()  # 确保立即写入磁盘
                    
                    
                    if 'wandb_data' in result:
                        wandb.log(result['wandb_data'])
                        
                except queue.Empty:
                    
                    continue
                except Exception as e:
                    print(f"Writer process error: {e}")
    
    
    def process_data_chunk(self, data_chunk, output_path, result_queue, worker_id, existing_results=None):
        """
        Process a subset of the dataset
        
        Args:
            data_chunk: data subset
            output_path: output directory
            result_queue: result queue for passing results to writer process
            worker_id: worker process ID
            existing_results: existing results to skip processed records
        """
        total_count = 0
        correct_count = 0
        all_tokens = 0
        
        
        pbar = tqdm(total=len(data_chunk), desc=f"Worker {worker_id}, model: {self.model}")
        if 'CoVR' in self.model:
            description_df = pd.read_csv('MAC_Bench/image2text_option_descriptions.csv')
        else:
            description_df = None
        for _, row in data_chunk.iterrows():
            journal = row['journal']
            question_id = row['id']
            question = row['question']
            image = row['cover_image']
            ground_truth = row['answer']
            
            if existing_results is not None and any(
                r['journal'] == journal and r['question_id'] == question_id 
                for r in existing_results
            ):
                pbar.update(1)
                continue
            
            
            if self.type == "image2text":
                
                options_list = []
                for opt_id in self.option_ids:
                    if f'option_{opt_id}' in row:
                        options_list.append(f'{opt_id}: {row[f"option_{opt_id}"]}')
                options = '\n'.join(options_list)
                messages = self.multi_choice_message(question, options, image)
                if 'CoVR' in self.model or 'majority' in self.model:
                    try:
                        
                        if self.descripted:
                            descriptions = description_df[(description_df['journal'] == journal) & (description_df['id'] == question_id)]['description'].values[0]
                            format_options = [(opt_id,row[f"option_{opt_id}"]) for opt_id in self.option_ids]
                            response,total_tokens, options = self.llm.post_existing_request(question=question,description=descriptions,options=format_options,messages= messages,type=self.type)
                        else:
                            format_options = [(opt_id,row[f"option_{opt_id}"]) for opt_id in self.option_ids]
                            response,total_tokens, options = self.llm.post_request(question=question,given_info=image,options=format_options,messages= messages,type=self.type)

                        
                    except Exception as e:
                        print(f"Error: LLM request failed and retries exhausted: {e}")
                        # answer = "None"
                        # judge = 0
                        # total_count += 1
                        # record = {
                        #     'journal': journal,
                        #     'question_id': question_id,
                        #     'question': question,
                        #     'options': options,
                        #     'ground_truth': ground_truth,
                        #     'response': response,
                        #     'total_tokens': total_tokens,
                        #     'answer': answer,
                        #     'judge': judge
                        # }
                        # all_tokens += total_tokens
                        # result_queue.put({
                        #     'record': record,
                        #     'wandb_data': {
                        #         "accuracy": correct_count/total_count, 
                        #         "all_tokens": all_tokens,
                        #         "num_options": self.num_options
                        #     }
                        # })
                        # pbar.update(1)
                        continue
                else:
                    try:
                        response, total_tokens = self.llm.post_request(messages=messages)
                    except Exception as e:
                        print(f"Error: LLM request failed and retries exhausted: {e}")
                        pbar.update(1)
                        continue
            elif self.type == "text2image":
                if 'CoVR' in self.model:
                    try:
                        format_options = [(opt_id,row[f"option_{opt_id}"]) for opt_id in self.option_ids]
                        response,total_tokens, options = self.llm.post_request(question=question,given_info=image,options=format_options,messages= ['test'],type=self.type)
                    except Exception as e:
                        print(f"Error: LLM request failed and retries exhausted: {e}")
                        continue
                else:   
                    try:
                        options_list = []
                        formatted_options = []
                        for opt_id in self.option_ids:
                            if f'option_{opt_id}' in row:
                                options_list.append(row[f"option_{opt_id}"])
                                formatted_options.append(f'{opt_id}: {row[f"option_{opt_id}"]}')

                        if 'ernie' in self.model:
                            low = True
                        else:
                            low = False
                        messages = self.multi_choice_message_text2image(question, image, options_list, low)
                        options = '\n'.join(formatted_options)
                        response, total_tokens = self.llm.post_request(
                            messages=messages  # here image is actually given_info, which is story
                        )
                    except Exception as e:
                        print(f"Error: LLM request failed and retries exhausted: {e}")
                        continue
            
            
            
            format_answer = parse_probabilities(response)
            if not format_answer:
                answer = "None"
                judge = 0
                total_count += 1
                # Create record
                record = {
                    'journal': journal,
                    'question_id': question_id,
                    'question': question,
                    'options': options,
                    'ground_truth': ground_truth,
                    'response': response,
                    'total_tokens': total_tokens,
                    'answer': answer,
                    'judge': judge
                }
                
                
                result_queue.put({
                    'record': record,
                    'wandb_data': {
                        "accuracy": correct_count/total_count, 
                        "all_tokens": all_tokens + total_tokens,
                        "num_options": self.num_options
                    }
                })
                
                all_tokens += total_tokens
                pbar.update(1)
                continue
            
            answer = max(format_answer, key=format_answer.get)
            total_count += 1
            judge = self.evaluation(ground_truth, answer)
            correct_count += judge
            
            # Create record
            record = {
                'journal': journal,
                'question_id': question_id,
                'question': question,
                'options': options,
                'ground_truth': ground_truth,
                'response': response,
                'total_tokens': total_tokens,
                'answer': answer,
                'judge': judge
            }
            
            
            result_queue.put({
                'record': record,
                'wandb_data': {
                    "accuracy": correct_count/total_count, 
                    "all_tokens": all_tokens + total_tokens,
                    "num_options": self.num_options
                }
            })
            
            all_tokens += total_tokens
            
            pbar.update(1)
        
        pbar.close()
        
        
        return {
            'worker_id': worker_id,
            'total_count': total_count,
            'correct_count': correct_count,
            'all_tokens': all_tokens
        }

    def experiment(self, data: pd.DataFrame, output_path: str, resume: bool = False, num_workers: int = 4):
        """
        Run experiment
        
        Args:
            data: dataset
            output_path: output directory
            resume: whether to continue previous experiment
            num_workers: number of parallel worker processes
        """
        
        os.makedirs(output_path, exist_ok=True)
        
        
        csv_path = os.path.join(output_path, 'results.csv')
        
        
        fieldnames = ['journal', 'question_id', 'question', 'options', 'ground_truth', 'response', 'total_tokens', 'answer', 'judge']
        
        
        existing_results = []
        if resume and os.path.exists(output_path):
            try:
                
                df = pd.read_csv(output_path, usecols=['journal', 'question_id'])
                existing_results = df.to_dict('records')
                print(f"Continuing previous experiment, already have {len(existing_results)} records")
            except Exception as e:
                print(f"Error reading CSV file: {e}")
                print("Trying alternative reading method...")
                
                with open(output_path, 'r', encoding='utf-8') as f:
                    
                    header = next(csv.reader(f))
                    journal_idx = header.index('journal')
                    question_id_idx = header.index('question_id')
                    
                    existing_results = []
                    for row in csv.reader(f):
                        existing_results.append({
                            'journal': row[journal_idx],
                            'question_id': row[question_id_idx]
                        })
                print(f"Successfully read using alternative method, already have {len(existing_results)} records")
            csv_path = output_path
        else:
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        
        
        result_queue = multiprocessing.Queue()
        
        
        stop_event = multiprocessing.Event()
        
        
        writer = multiprocessing.Process(
            target=self.writer_process,
            args=(csv_path, fieldnames, result_queue, stop_event),
            daemon=False 
        )
        writer.start()
        
        
        data_chunks = self.split_dataset(data, num_workers)
        
        
        processes = []
        stats = []
        
        for i, chunk in enumerate(data_chunks):
            p = multiprocessing.Process(
                target=self.process_data_chunk,
                args=(chunk, output_path, result_queue, i, existing_results),
                daemon=False
            )
            processes.append(p)
            p.start()
        
        
        for p in processes:
            p.join()
        
        
        stop_event.set()
        
        
        writer.join()
        
        print(f"Experiment completed, results saved to {csv_path}")

    def evaluation(self, ground_truth:str, answer:str):
        """
        评价答案
        """
        if ground_truth == answer:
            return 1
        else:
            return 0

    def experiment_with_threads(self, data: pd.DataFrame, output_path: str, resume: bool = False, num_workers: int = 4):
        """
        Use threads instead of processes to run experiment
        """
        
        if not os.path.exists(output_path) and not resume:
            os.makedirs(output_path, exist_ok=True)
            
            
            csv_path = os.path.join(output_path, 'results.csv')
            
        
        fieldnames = ['journal', 'question_id', 'question', 'options', 'ground_truth', 'response', 'total_tokens', 'answer', 'judge']
        
        
        existing_results = []
        if resume and os.path.exists(output_path):
            try:
                
                df = pd.read_csv(output_path, usecols=['journal', 'question_id'])
                existing_results = df.to_dict('records')
                print(f"Continuing previous experiment, already have {len(existing_results)} records")
            except Exception as e:
                print(f"Error reading CSV file: {e}")
                print("Trying alternative reading method...")
                
                with open(csv_path, 'r', encoding='utf-8') as f:
                    
                    header = next(csv.reader(f))
                    journal_idx = header.index('journal')
                    question_id_idx = header.index('question_id')
                    
                    existing_results = []
                    for row in csv.reader(f):
                        existing_results.append({
                            'journal': row[journal_idx],
                            'question_id': row[question_id_idx]
                        })
                print(f"Successfully read using alternative method, already have {len(existing_results)} records")
            csv_path = output_path
        else:
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        
        
        result_queue = queue.Queue()
        file_lock = threading.Lock()
        
        
        stop_event = threading.Event()
        writer_thread = threading.Thread(
            target=self.writer_thread,
            args=(csv_path, fieldnames, result_queue, stop_event, file_lock)
        )
        writer_thread.start()
        
        
        data_chunks = self.split_dataset(data, num_workers)
        
        
        threads = []
        for i, chunk in enumerate(data_chunks):
            t = threading.Thread(
                target=self.process_data_chunk,
                args=(chunk, output_path, result_queue, i, existing_results)
            )
            threads.append(t)
            t.start()
        
        
        for t in threads:
            t.join()
        
        
        stop_event.set()
        writer_thread.join()
        
        print(f"实验完成，结果已保存到 {csv_path}")

    def writer_thread(self, csv_path, fieldnames, result_queue, stop_event, file_lock):
        """
        A dedicated writing thread that reads results from the queue and writes to the CSV file
        
        Args:
            csv_path: CSV file path
            fieldnames: CSV file field names
            result_queue: Result queue
            stop_event: Stop event, used to notify the writing thread to end
            file_lock: Thread lock, used to synchronize file writing
        """
        
        csv.field_size_limit(2**27 - 1)  # Approximately 134,217,727
        
        while not stop_event.is_set() or not result_queue.empty():
            try:
                
                result = result_queue.get(timeout=1)
                
                
                with file_lock:
                    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(result['record'])
                        csvfile.flush()  # Ensure immediate disk write
                
                
                if 'wandb_data' in result:
                    wandb.log(result['wandb_data'])
                    
            except queue.Empty:
                
                continue
            except Exception as e:
                print(f"Writing thread error: {e}")
                print(f"Error details: {str(e)}")  # Add more detailed error information


def run_single_model_experiment(model_config, data, num_options, config, wandb_key):
    """
    运行单个模型的实验
    """
    try:
        
        test_model_api(model_config['name'], model_config['api_base'], model_config['api_key'])
        
        
        loader = ConfigLoader()
        prompt_template = loader.load_config(model_config['prompt_template'])
        
        
        wandb.login(key=wandb_key)
        run_name = f"Understanding_{model_config['name'].replace('/', '_')}_{config['data']['type']}_{config['data']['data_path'].replace('/', '_')}"
        wandb.init(project="CNS_cover", name=run_name)
        wandb.log({"model": model_config['name']})
        
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        
        experiment = MultiChoiceEvaluation(
            model=model_config['name'], 
            api_base=model_config['api_base'], 
            api_key=model_config['api_key'], 
            prompt_template=prompt_template,
            num_options=num_options,
            type=config['data']['type']
        )
        
        
        if model_config['resume']:
            output_path = model_config['resume_path']
            resume = True
        else:
            output_path = os.path.join(
                config['data']['output_folder'],
                f"{model_config['name'].replace('/','_')}_{num_options}options_{timestamp}"
            )
            resume = False
        
        
        num_workers = max(model_config.get('num_workers'),3)
        
        
        experiment.experiment_with_threads(
            data=data, 
            output_path=output_path, 
            resume=resume,
            num_workers=num_workers
        )
        
        
        wandb.finish()
        
        print(f"Model {model_config['name']} experiment completed, results saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Model {model_config['name']} experiment failed: {str(e)}")
        try:
            wandb.finish()
        except:
            pass
        return False

def run_models_in_batch(models_batch, data, num_options, config, wandb_key):
    """
    Run a batch of models
    """
    
    multiprocessing.freeze_support()
    
    
    pool = multiprocessing.Pool(processes=min(len(models_batch), 10))
    
    
    args_list = [(model, data, num_options, config, wandb_key) for model in models_batch]
    
    
    results = pool.starmap(run_single_model_experiment, args_list)
    
    
    pool.close()
    pool.join()
    
    
    return sum(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./Config/understanding_config.json", help="config file path")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of models to run per batch")
    args = parser.parse_args()
    
    loader = ConfigLoader()
    config = loader.load_config(args.config)
    models = config['models']
    
    
    num_options = config['data']['num_options']
    
    
    data = MultiChoiceEvaluation.prepare_dataset(
        data_path=config['data']['data_path'],
        scaling_factor=config['data']['scaling_factor'],
        seed=config['data']['random_seed']
    )
    
    
    wandb_key = "75c71a00697e97575abad4cafddb5cfc37de3305"
    
    
    batch_size = args.batch_size
    total_models = len(models)
    num_batches = math.ceil(total_models / batch_size)
    
    print(f"Split {total_models} models into {num_batches} batches, each batch has {batch_size} models")
    
    total_success = 0
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_models)
        current_batch = models[start_idx:end_idx]
        
        print(f"\nStart running batch {i+1}/{num_batches} ({len(current_batch)} models)")
        batch_success = run_models_in_batch(current_batch, data, num_options, config, wandb_key)
        total_success += batch_success
        
        print(f"Batch {i+1}/{num_batches} completed! Success: {batch_success}/{len(current_batch)}")
    
    print(f"\nAll batches completed! Total success: {total_success}/{total_models}")



