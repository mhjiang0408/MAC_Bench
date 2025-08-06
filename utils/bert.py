import torch
from transformers import AutoTokenizer, AutoModel
import os
import numpy as np

class BERTEmbedding:
    def __init__(self, model_name="bert-base-uncased"):
        """
        initialize BERT model for text embedding
        
        Args:
            model_name: BERT model name or path, default is "bert-base-uncased"
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        
        self.model.eval()
    
    def get_text_embedding(self, text, pooling_strategy="cls"):
        """
        get the embedding of a single text
        
        Args:
            text: text content
            pooling_strategy: pooling strategy, optional values are "cls"(use [CLS] token), "mean"(average all tokens) or "max"(take the maximum value of each dimension)
            
        Returns:
            numpy array, shape is (embedding_dim,)
        """
        
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            
            outputs = self.model(**inputs)
            
            
            last_hidden_state = outputs.last_hidden_state
            
            
            if pooling_strategy == "cls":
                
                embedding = last_hidden_state[:, 0, :]
            elif pooling_strategy == "mean":
                
                
                attention_mask = inputs["attention_mask"]
                embedding = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
            elif pooling_strategy == "max":
                
                attention_mask = inputs["attention_mask"]
                
                masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1) - (1 - attention_mask).unsqueeze(-1) * 1e10
                embedding = torch.max(masked_hidden, dim=1)[0]
            else:
                raise ValueError(f"不支持的池化策略: {pooling_strategy}")
            
            
            embedding = embedding.cpu().numpy()[0]
            
            return embedding
    
    def get_and_save_text_embedding(self, text, output_path, pooling_strategy="cls"):
        """
        get the embedding of a single text and save it to a file
        
        Args:
            text: text content
            output_path: output file path
            pooling_strategy: pooling strategy
            
        Returns:
            bool: True if successful, False if failed
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            
            embedding = self.get_text_embedding(text, pooling_strategy)
            
            
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
            torch.save(embedding_tensor, output_path)
            print(f"Text embedding saved to: {output_path}")
            
            return True
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return False
    
    def get_text_embeddings(self, texts, batch_size=32, pooling_strategy="cls"):
        """
        get the embedding of multiple texts
        
        Args:
            texts: text list
            batch_size: batch size
            pooling_strategy: pooling strategy
            
        Returns:
            numpy array, shape is (len(texts), embedding_dim)
        """
        all_embeddings = []
        
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            
            with torch.no_grad():
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                
                outputs = self.model(**inputs)
                
                
                last_hidden_state = outputs.last_hidden_state
                
                
                if pooling_strategy == "cls":
                    
                    embeddings = last_hidden_state[:, 0, :]
                elif pooling_strategy == "mean":
                    
                    attention_mask = inputs["attention_mask"]
                    embeddings = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
                elif pooling_strategy == "max":
                    
                    attention_mask = inputs["attention_mask"]
                    masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1) - (1 - attention_mask).unsqueeze(-1) * 1e10
                    embeddings = torch.max(masked_hidden, dim=1)[0]
                else:
                    raise ValueError(f"不支持的池化策略: {pooling_strategy}")
                
                
                batch_embeddings = embeddings.cpu().numpy()
                all_embeddings.append(batch_embeddings)
        
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    def eval_embedding_similarity(self, embedding1, embedding2, metric="cosine"):
        """
        calculate the similarity between two BERT embeddings
        
        Args:
            embedding1: first embedding, can be numpy array, torch tensor or list of them
            embedding2: second embedding, can be numpy array, torch tensor or list of them
            metric: similarity metric, optional values are "cosine"(cosine similarity), "euclidean"(Euclidean distance), "dot"(dot product)
                
        Returns:
            float or numpy array: if input is a single embedding, return float; if input is a list of embeddings, return similarity matrix
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
    
    def _compute_single_similarity(self, embedding1, embedding2, metric="cosine"):
        """
        calculate the similarity between two single embeddings
        
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
        
        elif metric == "euclidean":
            
            distance = np.linalg.norm(embedding1 - embedding2)
            
            return np.exp(-distance)
        
        elif metric == "dot":
            
            return np.dot(embedding1, embedding2)
        
        else:
            raise ValueError(f"Not supported similarity metric: {metric}")
