import open_clip
from PIL import Image
import torch
import numpy as np

class OpenClip:
    def __init__(self, model_name='ViT-H-14-quickgelu', pretrained='metaclip_fullcc'):
        self.model_name = model_name
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def save_image_embeddings(self, image_path, save_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.preprocess(image).unsqueeze(0)
            image_features = self.model.encode_image(image)
            torch.save(image_features.cpu(), save_path)
        except Exception as e:
            print(f"Error saving image embeddings: {e}")
            return False
        return True
    
    def save_text_embeddings(self, text, save_path):
        try:
            text = self.tokenizer(text)
            text_features = self.model.encode_text(text)
            torch.save(text_features.cpu(), save_path)
        except Exception as e:
            print(f"Error saving text embeddings: {e}")
            return False
        return True
    
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
    