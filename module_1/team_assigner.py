import torch
import numpy as np
import supervision as sv
from transformers import AutoImageProcessor, SiglipVisionModel
import umap.umap_ as umap
from sklearn.cluster import KMeans
from typing import List

class TeamAssigner:
    def __init__(self, model_name: str, n_teams: int, n_components: int):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SiglipVisionModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.reducer = umap.UMAP(n_components=n_components)
        self.cluster_model = KMeans(n_clusters=n_teams, n_init="auto")
        self.is_trained = False
        
    def extract_features(self, crops: List[np.ndarray], batch_size: int = 16) -> np.ndarray:
        """Trích xuất vector đặc trưng và ép giải phóng VRAM."""
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(crops), batch_size):
                batch = crops[i:i + batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                all_embeddings.append(embeddings)
                
        # Bắt buộc dọn cache VRAM sau khi inference batch
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            
        return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, 768))

    def fit(self, crops: List[np.ndarray]):
        """Train UMAP và KMeans."""
        if not crops:
            raise ValueError("Không có crop data để train.")
        embeddings = self.extract_features(crops)
        reduced_data = self.reducer.fit_transform(embeddings)
        self.cluster_model.fit(reduced_data)
        self.is_trained = True
        
    def predict(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Dự đoán đội (0 hoặc 1) cho các bounding box trong frame."""
        if not self.is_trained or len(detections.xyxy) == 0:
            return np.zeros(len(detections.xyxy), dtype=int)
            
        crops = [sv.cv2_to_pillow(sv.crop_image(frame, box)) for box in detections.xyxy]
        embeddings = self.extract_features(crops)
        reduced_data = self.reducer.transform(embeddings)
        return self.cluster_model.predict(reduced_data)