from sentence_transformers import SentenceTransformer,util
import torch
import numpy as np
class SBERT_Embedding:
    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, text:str):
        return self.model.encode(text, convert_to_tensor=True)
    
    def get_and_save_text_embedding(self, text:str, output_path:str):
        try:
            embedding = self.encode(text)
            torch.save(embedding, output_path)
            print(f"text embedding saved to: {output_path}")
            return True
        except Exception as e:
            print(f"error processing text: {str(e)}")
            return False
    def _compute_single_similarity(self, embedding1, embedding2, metric="cosine"):
        """
        compute similarity between two single embeddings
        
        Args:
            embedding1: first embedding, numpy array or torch tensor
            embedding2: second embedding, numpy array or torch tensor
            metric: similarity metric
                
        Returns:
            float: similarity score
        """
        
        if isinstance(embedding1, torch.Tensor):
            embedding1 = embedding1.cpu().numpy()
        if isinstance(embedding2, torch.Tensor):
            embedding2 = embedding2.cpu().numpy()
        
        
        if len(embedding1.shape) > 1 and embedding1.shape[0] == 1:
            embedding1 = embedding1.flatten()
        if len(embedding2.shape) > 1 and embedding2.shape[0] == 1:
            embedding2 = embedding2.flatten()
        
        
        if metric == "cosine":
            
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    def eval_embedding_similarity(self, embedding1, embedding2, metric="cosine"):
        """
        compute similarity between two BERT embeddings
        
        Args:
            embedding1: first embedding, can be numpy array, torch tensor or list of them
            embedding2: second embedding, can be numpy array, torch tensor or list of them
            metric: similarity metric, can be "cosine"(cosine similarity),
                "euclidean"(Euclidean distance), "dot"(dot product)
                
        Returns:
            float or numpy array: if input is a single embedding, return float;
                                 if input is a list of embeddings, return similarity matrix
        """
        
        is_list1 = isinstance(embedding1, list)
        is_list2 = isinstance(embedding2, list)
        
        
        if not is_list1 and not is_list2:
            return self._compute_single_similarity(embedding1, embedding2, metric)
        
        
        if is_list1 and not is_list2:
            embedding2 = [embedding2]
        elif not is_list1 and is_list2:
            embedding1 = [embedding1]
        
        
        similarity_matrix = np.zeros((len(embedding1), len(embedding2)))
        for i, emb1 in enumerate(embedding1):
            for j, emb2 in enumerate(embedding2):
                similarity_matrix[i, j] = self._compute_single_similarity(emb1, emb2, metric)
        
        return similarity_matrix

