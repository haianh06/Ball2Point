import os
from pathlib import Path
from dataclasses import dataclass
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class HeatmapConfig:
    # Alpha blending factor for overlaying heatmap on original frame (0.0 - only heatmap, 1.0 - only original)
    ALPHA_BLEND: float = 0.65  
    
    # Colormap to use for heatmap visualization. JET is a common choice for intensity maps, but can be changed as needed.
    COLORMAP = cv2.COLORMAP_JET
    
    # Gaussian kernel size for smoothing the heatmap. Larger kernels will produce smoother heatmaps but may blur details. Must be odd numbers.
    GAUSSIAN_KERNEL: tuple = (91, 91)  
    
    OUTPUT_DIR: str = str(PROJECT_ROOT / "outputs/player_heatmaps")

config = HeatmapConfig()