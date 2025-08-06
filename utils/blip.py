import torch
import torch.nn.functional as F
from transformers import Blip2Processor, Blip2Model
from PIL import Image

class BLIP2Extractor:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device="cuda"):
        """
        initialize BLIP-2 feature extractor
        """
        self.device = device
        self.processor = Blip2Processor.from_pretrained(model_name)
        
        self.model = Blip2Model.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(device)
        self.model.eval()

    def extract_features(self, image_path=None, text=None):
        """
        extract features from image or text
        
        Args:
            image_path: image path (optional)
            text: text content (optional)
            
        Returns:
            torch.Tensor: feature vector
        """
        if image_path is not None:
            
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
            with torch.no_grad():
                outputs = self.model.vision_model(**inputs)
                
                image_features = outputs.pooler_output
                
                image_features = F.normalize(image_features, dim=-1)
                return image_features
                
        elif text is not None:
            
            inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device, torch.float16)
            with torch.no_grad():
                outputs = self.model.language_model(**inputs)
                
                text_features = outputs.last_hidden_state.mean(dim=1)
                
                text_features = F.normalize(text_features, dim=-1)
                return text_features
        
        else:
            raise ValueError("Must provide image path or text")

    def compute_similarity(self, image_path, text):
        """
        calculate the similarity between image and text
        
        Args:
            image_path: image path
            text: text content
            
        Returns:
            float: similarity score
        """
        
        image_features = self.extract_features(image_path=image_path)
        text_features = self.extract_features(text=text)
        
        
        similarity = torch.mm(image_features, text_features.transpose(0, 1))
        return similarity.item()

def main():
    
    extractor = BLIP2Extractor()
    
    
    image_path = "./Cell/Cover/Biophysical Reports/1_1.png"
    text = """On the cover: Exchangeable dyes enable stimulated emission depletion (STED) imaging of lipid exchange
during membrane fusion. STED super-resolution microscopy can induce photobleaching
that limits long-term sample observation. To circumvent this problem, Carravilla et
al. use exchangeable dyes that only temporarily bind to their target. A recently developed
polarity-sensitive exchangeable plasma membrane probe based on Nile Red permits the
super-resolved quantification of membrane biophysical parameters in real time with
high temporal and spatial resolution and long acquisition times."""
    
    
    similarity = extractor.compute_similarity(image_path, text)
    print(f"Image-Text Similarity: {similarity:.4f}")
    
    image_features = extractor.extract_features(image_path=image_path)
    text_features = extractor.extract_features(text=text)
    print("Image feature shape:", image_features.shape)
    print("Text feature shape:", text_features.shape)

if __name__ == "__main__":
    main()