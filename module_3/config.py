import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class SpeedConfig:
    KEYPOINT_MODEL_PATH: str = os.getenv("POSE_MODEL_PATH", str(PROJECT_ROOT / "Models/weights/best_keypoints.pt"))
    CONFIDENCE_THRESHOLD: float = 0.5
    
    WINDOW_SIZE: int = 5  # Calculate speed over the last 5 frames
    STATS_OUTPUT_PATH: str = os.getenv("STATS_OUTPUT_PATH", str(PROJECT_ROOT / "outputs/player_statistics.json"))
    
    # Fallback pixel-to-meter ratio if homography fails (e.g., no field detected)
    PIXEL_TO_METER_FALLBACK: float = 0.05
    
    @staticmethod
    def get_standard_pitch_points() -> dict:
        """Standard FIFA meter coordinates for Homography calculation."""
        return {
            0: [0.0, 0.0],                          1: [0.0, 13.85],
            2: [16.5, 13.85],                       3: [0.0, 54.15],
            4: [16.5, 54.15],                       5: [0.0, 24.85],
            6: [5.5, 24.85],                        7: [0.0, 43.15],
            8: [5.5, 43.15],                        9: [0.0, 68.0],
            10: [16.5, 34.0],                       11: [52.5, 0.0],
            12: [52.5, 68.0],                       13: [52.5, 24.85],
            14: [52.5, 43.15],                      15: [52.5, 34.0],
            16: [105.0, 0.0],                       17: [105.0, 13.85],
            18: [88.5, 13.85],                      19: [105.0, 54.15],
            20: [88.5, 54.15],                      21: [105.0, 24.85],
            22: [99.5, 24.85],                      23: [105.0, 43.15],
            24: [99.5, 43.15],                      25: [105.0, 68.0],
            26: [88.5, 34.0],                       27: [43.35, 34.0],
            28: [61.65, 34.0],
        }

config = SpeedConfig()