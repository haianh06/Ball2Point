import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class ModelConfig:
    # Sử dụng os.path.normpath để xử lý dấu gạch chéo ngược của Windows
    DETECTION_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", str(PROJECT_ROOT / "Models/weights/best_detection_v2.pt"))
    SIGLIP_MODEL_NAME: str = "google/siglip-base-patch16-224"
    
    # Thêm giá trị default để tránh lỗi NoneType khi gọi OpenCV
    INPUT_VIDEO_PATH: str = os.getenv("INPUT_VIDEO_PATH", "inputs/test_1.mp4")
    OUTPUT_VIDEO_PATH: str = os.getenv("OUTPUT_VIDEO_PATH", "outputs/result_1.avi")

    # Tracking & Clustering Params
    TRACKER_MATCH_THRESH: float = 0.5
    TRACKER_BUFFER: int = 120
    N_TEAMS: int = 2
    UMAP_COMPONENTS: int = 3
    EMBEDDING_BATCH_SIZE: int = 16 

config = ModelConfig()