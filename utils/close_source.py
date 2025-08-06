from curl_cffi import requests
import dashscope
import json
from http import HTTPStatus
import torch
import time
import io
import base64
from PIL import Image


class QwenEmbedding:
    def __init__(self,model_name="multimodal-embedding-v1"):
        self.model = model_name
        self.api_key = "sk-xxx"
    def save_image_embeddings(self, image_path,save_path):
        # image_base64 = base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')
        img = Image.open(image_path)
        img_byte_arr = io.BytesIO()
        img.convert('RGB').save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        
        image_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
        input = [{'image': f"data:image/jpeg;base64,{image_base64}"}]

        resp = dashscope.MultiModalEmbedding.call(
            model=self.model,
            input=input,
            api_key=self.api_key,
        )
        if resp.status_code == HTTPStatus.OK:
            embedding = torch.tensor(resp.output["embeddings"][0]['embedding'])
            torch.save(embedding, save_path)
            time.sleep(2)
            return True
        else:
            print(f"Failed to get embedding for image {image_path}, status code: {resp}")
            return False
    
    def save_text_embeddings(self, text,save_path):
        text = text.strip()
        text = text.replace("\n", "")
        text = ' '.join(text.split())
        input = [{'text': text}]

        resp = dashscope.MultiModalEmbedding.call(
            model=self.model,
            input=input,
            api_key=self.api_key,
        )
        if resp.status_code == HTTPStatus.OK:
            embedding = torch.tensor(resp.output["embeddings"][0]['embedding'])
            torch.save(embedding, save_path)
            time.sleep(1)
            return True
        else:
            print(f"Failed to get embedding for text {text}")
            return False
    def eval_embedding_similarity(self, embeddings1, embeddings2):
        """
        load saved embeddings and evaluate the similarity between two embedding lists
        
        Args:
            image_embeddings_paths: image embedding file path list, optional, if image_embeddings is not None
            text_embeddings_paths: text embedding file path list, optional, if text_embeddings is not None
            image_embeddings: loaded image embedding list, optional, if image_embeddings_paths is not None
            text_embeddings: loaded text embedding list, optional, if text_embeddings_paths is not None
            normalize: whether to normalize the embeddings
            return_probs: whether to return probability distribution (using softmax) instead of raw similarity
            
        Returns:
            numpy.ndarray: similarity matrix, shape [num_images, num_texts]
            if return_probs=True, return probability distribution
        """
        
        # if image_embeddings is None and image_embeddings_paths is not None:
        #     image_embeddings = []
        #     for path in image_embeddings_paths:
        #         try:
        #             emb = torch.load(path)
        #             image_embeddings.append(emb)
        #         except Exception as e:
        #             print(f"Error loading image embedding from {path}: {e}")
        #             continue
            
        #     if not image_embeddings:
        #         raise ValueError("No valid image embeddings could be loaded")
            
        
        #     image_embeddings = torch.cat(image_embeddings, dim=0)
        
        
        # if text_embeddings is None and text_embeddings_paths is not None:
        #     text_embeddings = []
        #     for path in text_embeddings_paths:
        #         try:
        #             emb = torch.load(path)
        #             text_embeddings.append(emb)
        #         except Exception as e:
        #             print(f"Error loading text embedding from {path}: {e}")
        #             continue
            
        #     if not text_embeddings:
        #         raise ValueError("No valid text embeddings could be loaded")
            
        
        #     text_embeddings = torch.cat(text_embeddings, dim=0)
        
        
        if embeddings1 is None or embeddings2 is None:
            raise ValueError("Either provide embedding paths or pre-loaded embeddings")
        if isinstance(embeddings1, list):
            embeddings1 = torch.cat(embeddings1, dim=0)
        if isinstance(embeddings2, list):
            embeddings2 = torch.cat(embeddings2, dim=0)
        
        
        if not isinstance(embeddings1, torch.Tensor):
            embeddings1 = torch.tensor(embeddings1)
        if not isinstance(embeddings2, torch.Tensor):
            embeddings2 = torch.tensor(embeddings2)
        
        
        embeddings1 = embeddings1 / embeddings1.norm(dim=-1, keepdim=True)
        embeddings2 = embeddings2 / embeddings2.norm(dim=-1, keepdim=True)
        
        
        with torch.no_grad():
            similarity = (embeddings1 @ embeddings2.T).cpu()
            
            
            # if return_probs:
            #     similarity = (100.0 * similarity).softmax(dim=-1).numpy()
            # else:
            similarity = similarity.numpy()
        
        return similarity


class DoubaoEmbedding:
    def __init__(self,model_name="multimodal-embedding-v1"):
        self.api_base = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
        self.model = model_name
        self.api_key = "1f8efed2-5349-4bec-aa8a-948b6f1bbf5c"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer 1f8efed2-5349-4bec-aa8a-948b6f1bbf5c',
        }
    def save_image_embeddings(self, image_path,save_path):
        image_base64 = base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')
        json_data = {
            'model': 'doubao-embedding-vision-241215',
            'input': [
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f"data:image/png;base64,{image_base64}",
                    }
                }
            ],
        }
        try:
            response = requests.post(self.api_base, headers=self.headers, json=json_data)
            embedding = torch.tensor(response.json()['data']['embedding'])
            torch.save(embedding, save_path)
            return True
        except Exception as e:
            print(f"Failed to get embedding for image {image_path}: {e}")
            return False
        
    def save_text_embeddings(self, text,save_path):
        text = text.strip()
        text = text.replace("\n", "")
        text = ' '.join(text.split())
        json_data = {
            'model': 'doubao-embedding-vision-241215',
            'input': [
                {
                    'type': 'text',
                    'text': text
                }
            ]
        }
        try:
            response = requests.post(self.api_base, headers=self.headers, json=json_data)
            embedding = torch.tensor(response.json()['data']['embedding'])
            torch.save(embedding, save_path)
            return True
        except Exception as e:
            print(f"Failed to get embedding for text {text}: {e}")
            return False
    
    def eval_embedding_similarity(self, embeddings1, embeddings2):
        """
        load saved embeddings and evaluate the similarity between two embedding lists
        
        Args:
            image_embeddings_paths: image embedding file path list, optional, if image_embeddings is not None
            text_embeddings_paths: text embedding file path list, optional, if text_embeddings is not None
            image_embeddings: loaded image embedding list, optional, if image_embeddings_paths is not None
            text_embeddings: loaded text embedding list, optional, if text_embeddings_paths is not None
            normalize: whether to normalize the embeddings
            return_probs: whether to return probability distribution (using softmax) instead of raw similarity
            
        Returns:
            numpy.ndarray: similarity matrix, shape [num_images, num_texts]
            if return_probs=True, return probability distribution
        """
        
        # if image_embeddings is None and image_embeddings_paths is not None:
        #     image_embeddings = []
        #     for path in image_embeddings_paths:
        #         try:
        #             emb = torch.load(path)
        #             image_embeddings.append(emb)
        #         except Exception as e:
        #             print(f"Error loading image embedding from {path}: {e}")
        #             continue
            
        #     if not image_embeddings:
        #         raise ValueError("No valid image embeddings could be loaded")
            
        
        #     image_embeddings = torch.cat(image_embeddings, dim=0)
        
        
        # if text_embeddings is None and text_embeddings_paths is not None:
        #     text_embeddings = []
        #     for path in text_embeddings_paths:
        #         try:
        #             emb = torch.load(path)
        #             text_embeddings.append(emb)
        #         except Exception as e:
        #             print(f"Error loading text embedding from {path}: {e}")
        #             continue
            
        #     if not text_embeddings:
        #         raise ValueError("No valid text embeddings could be loaded")
            
        
        #     text_embeddings = torch.cat(text_embeddings, dim=0)
        
        
        if embeddings1 is None or embeddings2 is None:
            raise ValueError("Either provide embedding paths or pre-loaded embeddings")
        if isinstance(embeddings1, list):
            embeddings1 = torch.cat(embeddings1, dim=0)
        if isinstance(embeddings2, list):
            embeddings2 = torch.cat(embeddings2, dim=0)
        
        
        if not isinstance(embeddings1, torch.Tensor):
            embeddings1 = torch.tensor(embeddings1)
        if not isinstance(embeddings2, torch.Tensor):
            embeddings2 = torch.tensor(embeddings2)
        
        
        embeddings1 = embeddings1 / embeddings1.norm(dim=-1, keepdim=True)
        embeddings2 = embeddings2 / embeddings2.norm(dim=-1, keepdim=True)
        
        
        with torch.no_grad():
            similarity = (embeddings1 @ embeddings2.T).cpu()
            
            # if return_probs:
            #     similarity = (100.0 * similarity).softmax(dim=-1).numpy()
            # else:
            similarity = similarity.numpy()
        
        return similarity