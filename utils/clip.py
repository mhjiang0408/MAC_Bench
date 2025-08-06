import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import os
import json
from PIL import Image
import pandas as pd

class CLIPEmbedding:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        initialize CLIP model for text and image embedding
        
        Args:
            model_name: CLIP model name or path
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        
        self.model.eval()
        
    def get_text_embeddings(self, texts, batch_size=32):
        """
        get the embedding of texts
        
        Args:
            texts: text list
            batch_size: batch size
            
        Returns:
            numpy array, shape is (len(texts), embedding_dim)
        """
        all_embeddings = []
        
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            
            with torch.no_grad():
                inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                
                text_features = self.model.get_text_features(**inputs)
                
                
                text_embeddings = text_features / text_features.norm(dim=1, keepdim=True)
                
                
                all_embeddings.append(text_embeddings.cpu().numpy())
        
        
        return np.vstack(all_embeddings)
    
    def get_and_save_text_embedding(self, text, output_path):
        """
        convert a single text to embedding and save it
        
        Args:
            text: text content
            output_path: path to save embedding
            
        Returns:
            bool: True if successful, False if failed
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            
            with torch.no_grad():
                inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                
                text_features = self.model.get_text_features(**inputs)
                
                
                text_embedding = text_features / text_features.norm(dim=1, keepdim=True)
                
                
                torch.save(text_embedding[0].cpu().float(), output_path)
                print(f"Text embedding saved to: {output_path}")
                
                return True
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return False

    def get_and_save_image_embedding(self, image_path, output_path):
        """
        convert a single image to embedding and save it
        
        Args:
            image_path: image file path or PIL.Image object
            output_path: path to save embedding
            
        Returns:
            bool: True if successful, False if failed
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            
            if isinstance(image_path, str):
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    print(f"Error loading image {image_path}: {str(e)}")
                    return False
            else:
                
                image = image_path
                
            
            with torch.no_grad():
                inputs = self.processor(images=[image], return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                
                image_features = self.model.get_image_features(**inputs)
                
                
                image_embedding = image_features / image_features.norm(dim=1, keepdim=True)
                
                
                torch.save(image_embedding[0].cpu(), output_path)
                print(f"Image embedding saved to: {output_path}")
                
                return True
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return False
    
    def get_image_embeddings(self, images, batch_size=16):
        """
        get the embedding of images
        
        Args:
            images: image path list or PIL.Image object list
            batch_size: batch size
            
        Returns:
            numpy array, shape is (len(images), embedding_dim)
        """
        all_embeddings = []
        
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            processed_images = []
            
            
            for img in batch_images:
                if isinstance(img, str):  # if it is a path
                    try:
                        img = Image.open(img).convert('RGB')
                    except Exception as e:
                        print(f"Error loading image {img}: {e}")
                        continue
                processed_images.append(img)
            
            if not processed_images:
                continue
                
            
            with torch.no_grad():
                inputs = self.processor(images=processed_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                
                image_features = self.model.get_image_features(**inputs)
                
                
                image_embeddings = image_features / image_features.norm(dim=1, keepdim=True)
                
                
                all_embeddings.append(image_embeddings.cpu().numpy())
        
        if not all_embeddings:
            return np.array([])
            
        
        return np.vstack(all_embeddings)
    
    def eval_similarity(self, text1, text2=None, images=None):
        """
        calculate the similarity between text and text or text and image
        
        Args:
            text1: text1 or text list1
            text2: text2 or text list2, if compared with image, it is None
            images: image path or PIL.Image object list, if compared with text, it is None
            
        Returns:
            similarity score (between 0 and 1), if input is a list, return similarity matrix
        """
        
        if isinstance(text1, str):
            text1 = [text1]
        
        
        embeddings1 = self.get_text_embeddings(text1)
        
        
        if text2 is not None:
            if isinstance(text2, str):
                text2 = [text2]
            embeddings2 = self.get_text_embeddings(text2)
        
        elif images is not None:
            if isinstance(images, str) or isinstance(images, Image.Image):
                images = [images]
            embeddings2 = self.get_image_embeddings(images)
        else:
            raise ValueError("Either text2 or images must be provided")
        
        
        similarity_matrix = np.matmul(embeddings1, embeddings2.T)
        
        
        if len(embeddings1) == 1 and len(embeddings2) == 1:
            return float(similarity_matrix[0, 0])
        
        return similarity_matrix

    def eval_embedding_similarity(self, embeddings1, embeddings2):
        """
        calculate the similarity between two groups of embeddings, support embedding list as input
        
        Args:
            embeddings1: first group of embeddings, can be:
                        - single numpy array or torch tensor
                        - list of numpy array or torch tensor
            embeddings2: second group of embeddings, format same as embeddings1
                
        Returns:
            similarity matrix or single similarity value
        """
        
        if isinstance(embeddings1, (list, tuple)):
            
            if all(isinstance(e, torch.Tensor) for e in embeddings1):
                
                normalized_embeddings = []
                for emb in embeddings1:
                    emb = emb.float()
                    
                    if len(emb.shape) == 1:
                        emb = emb.unsqueeze(0)
                    
                    emb = emb / emb.norm(dim=1, keepdim=True)
                    normalized_embeddings.append(emb)
                
                
                embeddings1 = torch.cat(normalized_embeddings, dim=0)
            elif all(isinstance(e, np.ndarray) for e in embeddings1):
                
                normalized_embeddings = []
                for emb in embeddings1:
                    if len(emb.shape) == 1:
                        emb = emb.reshape(1, -1)
                    
                    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                    normalized_embeddings.append(emb)
                
                
                embeddings1 = np.vstack(normalized_embeddings)
        
        
        if isinstance(embeddings2, (list, tuple)):
            
            if all(isinstance(e, torch.Tensor) for e in embeddings2):
                
                normalized_embeddings = []
                for emb in embeddings2:
                    emb = emb.float()
                    
                    if len(emb.shape) == 1:
                        emb = emb.unsqueeze(0)
                    
                    emb = emb / emb.norm(dim=1, keepdim=True)
                    normalized_embeddings.append(emb)
                
                
                embeddings2 = torch.cat(normalized_embeddings, dim=0)
            elif all(isinstance(e, np.ndarray) for e in embeddings2):
                
                normalized_embeddings = []
                for emb in embeddings2:
                    if len(emb.shape) == 1:
                        emb = emb.reshape(1, -1)
                    
                    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                    normalized_embeddings.append(emb)
                
                
                embeddings2 = np.vstack(normalized_embeddings)
        
        
        
        if isinstance(embeddings1, torch.Tensor):
            embeddings1 = embeddings1.float()
            if len(embeddings1.shape) == 1:
                embeddings1 = embeddings1.unsqueeze(0)
            embeddings1 = embeddings1 / embeddings1.norm(dim=1, keepdim=True)
        
        if isinstance(embeddings2, torch.Tensor):
            embeddings2 = embeddings2.float()
            if len(embeddings2.shape) == 1:
                embeddings2 = embeddings2.unsqueeze(0)
            embeddings2 = embeddings2 / embeddings2.norm(dim=1, keepdim=True)
        
        
        if isinstance(embeddings1, torch.Tensor):
            embeddings1 = embeddings1.cpu().numpy()
        if isinstance(embeddings2, torch.Tensor):
            embeddings2 = embeddings2.cpu().numpy()
        
        
        if len(embeddings1.shape) == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if len(embeddings2.shape) == 1:
            embeddings2 = embeddings2.reshape(1, -1)
        
        
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        
        similarity_matrix = np.matmul(embeddings1, embeddings2.T)
        
        
        if embeddings1.shape[0] == 1 and embeddings2.shape[0] == 1:
            return float(similarity_matrix[0, 0])
        
        return similarity_matrix
    
    def find_best_matches(self, query, candidates, is_image_query=False, top_k=5):
        """
        find the best matches for the query
        
        Args:
            query: query text or image path
            candidates: candidate text list or image path list
            is_image_query: whether the query is an image
            top_k: number of best matches to return
            
        Returns:
            list of (index, similarity, candidate), sorted by similarity in descending order
        """
        if is_image_query:
            
            if isinstance(query, str):
                query = [query]  # convert to list for processing
            query_embeddings = self.get_image_embeddings(query)
            candidate_embeddings = self.get_text_embeddings(candidates)
        else:
            
            if isinstance(query, str):
                query = [query]
            query_embeddings = self.get_text_embeddings(query)
            
            
            if all(isinstance(c, str) and (c.endswith('.jpg') or c.endswith('.png') or c.endswith('.jpeg')) for c in candidates):
                
                candidate_embeddings = self.get_image_embeddings(candidates)
            else:
                
                candidate_embeddings = self.get_text_embeddings(candidates)
        
        
        similarity = np.matmul(query_embeddings, candidate_embeddings.T)[0]  # 取第一个查询的结果
        
        
        top_indices = similarity.argsort()[-top_k:][::-1]
        
        
        results = []
        for idx in top_indices:
            results.append((int(idx), float(similarity[idx]), candidates[idx]))
        
        return results

    def save_embeddings(self, items, embeddings, output_path, is_image=False):
        """
        save text/image and corresponding embeddings
        
        Args:
            items: text list or image path list
            embeddings: embedding array
            output_path: output file path
            is_image: whether the embeddings are image embeddings
        """
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        
        if output_path.endswith('.npy'):
            np.save(output_path, embeddings)
            
            items_path = output_path.replace('.npy', '_items.json')
            with open(items_path, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            print(f"Embeddings saved to {output_path}")
            print(f"Items saved to {items_path}")
            
        
        elif output_path.endswith('.csv'):
            
            df = pd.DataFrame({
                'item': items,
                'is_image': is_image
            })
            
            
            for i in range(embeddings.shape[1]):
                df[f'embedding_{i}'] = embeddings[:, i]
                
            df.to_csv(output_path, index=False)
            print(f"Embeddings and items saved to {output_path}")
            
        else:
            raise ValueError("Output path must end with .npy or .csv")