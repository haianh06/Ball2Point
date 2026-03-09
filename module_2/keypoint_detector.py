from ultralytics import YOLO
import numpy as np
import torch

class KeypointDetector:
    def __init__(self, model_path: str, conf_threshold: float):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = '0' if torch.cuda.is_available() else 'cpu'

    def detect(self, frame: np.ndarray) -> np.ndarray:
        # Ép chạy trên GPU bằng tham số device
        results = self.model(frame, device=self.device, verbose=False)[0]
        if hasattr(results, 'keypoints') and results.keypoints is not None:
            kpts = results.keypoints.data.cpu().numpy()
            if len(kpts) > 0:
                return kpts[0]
        return np.array([])