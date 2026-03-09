from dataclasses import dataclass

@dataclass
class VoronoiConfig:
    ALPHA_BLEND: float = 0.35  
    
    TEAM_0_COLOR: tuple = (255, 0, 0)
    TEAM_1_COLOR: tuple = (0, 0, 255)
    
    DRAW_BORDERS: bool = True
    BORDER_COLOR: tuple = (255, 255, 255)
    
    REAL_LENGTH: float = 105.0
    REAL_WIDTH: float = 68.0

config = VoronoiConfig()