from ultralytics import YOLO
import supervision as sv
import numpy as np
from typing import Tuple
import torch

class SoccerDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.device = '0' if torch.cuda.is_available() else 'cpu'

    def detect(self, frame: np.ndarray) -> Tuple[sv.Detections, sv.Detections, sv.Detections]:
        """Return: (players, ball, referees)"""
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        players = detections[detections.class_id == 0]
        ball = detections[detections.class_id == 1]
        referees = detections[detections.class_id == 2]
        
        return players, ball, referees