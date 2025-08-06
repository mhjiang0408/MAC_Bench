import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import requests
from typing import Union, List
import numpy as np
from transformers.image_utils import load_image

class SigLIP2Encoder:
    def __init__(self, model_name: str = "google/siglip2-so400m-patch16-naflex"):
        """
        initialize SigLIP2 encoder
        
        Args:
            model_name: model name, default is base version
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        
    def save_image_embeddings(self, images: Union[str, List[str], Image.Image, List[Image.Image]], save_path: str) -> np.ndarray:
        """
        encode images
        
        Args:
            images: can be image path, PIL image object or list of them
            
        Returns:
            image feature vector
        """
        
        try:
            if not isinstance(images, list):
                images = [images]
                
            
            pil_images = []
            for image in images:
                pil_images.append(load_image(image))
                    
            
            inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
            
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs).cpu()
                
            
            torch.save(image_features.cpu(), save_path)
            return True
        except Exception as e:
            print(f"Error saving image embeddings: {e}")
            return False
    
    def save_text_embeddings(self, texts: Union[str, List[str]], save_path: str) -> np.ndarray:
        """
        encode texts
        
        Args:
            texts: text or list of texts
            
        Returns:
            text feature vector
        """
        
        try:
            if isinstance(texts, str):
                texts = [texts]
                
            
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            torch.save(text_features.cpu(), save_path)
            return True
        except Exception as e:
            print(f"Error saving text embeddings: {e}")
            return False
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # return text_features.cpu().numpy()
    
    def eval_embedding_similarity(self, embeddings1, embeddings2):
        """
        load saved embeddings and evaluate similarity between two embedding lists
        
        Args:
            image_embeddings_paths: image embedding file path list, optional
            text_embeddings_paths: text embedding file path list, optional
            image_embeddings: loaded image embedding list, optional
            text_embeddings: loaded text embedding list, optional
            normalize: whether to normalize embeddings
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