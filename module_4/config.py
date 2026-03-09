import os
from pathlib import Path
from dataclasses import dataclass
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class HeatmapConfig:
    # Hệ số trong suốt khi đè lên sân cỏ (0.0 -> 1.0)
    ALPHA_BLEND: float = 0.65  
    
    # Bảng màu (Jet: Đỏ là nóng nhất, Xanh là lạnh)
    COLORMAP = cv2.COLORMAP_JET
    
    # Độ lan tỏa của nhiệt. Tăng số này (vd: 121, 151) nếu muốn vệt đỏ to hơn
    GAUSSIAN_KERNEL: tuple = (91, 91)  
    
    # Thư mục xuất ảnh đầu ra
    OUTPUT_DIR: str = str(PROJECT_ROOT / "outputs/player_heatmaps")

config = HeatmapConfig()