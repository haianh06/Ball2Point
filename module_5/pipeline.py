import numpy as np
from .config import config
from .voronoi_engine import VoronoiEngine

class PitchControlPipeline:
    def __init__(self):
        self.engine = VoronoiEngine()
        self.last_pixel_points = {}

    def _real_to_pixel(self, x_real: float, y_real: float, img_width: int, img_height: int) -> tuple:
        """Scale real-world coordinates to pixel coordinates based on the pitch image dimensions and margins."""
        margin_x = int(img_width * 0.046)
        margin_y = int(img_height * 0.046)
        draw_width = img_width - 2 * margin_x
        draw_height = img_height - 2 * margin_y
        
        x_pct = x_real / config.REAL_LENGTH
        y_pct = y_real / config.REAL_WIDTH
        
        x_px = margin_x + int(x_pct * draw_width)
        y_px = margin_y + int(y_pct * draw_height)
        return x_px, y_px

    def process_frame(self, tactical_canvas: np.ndarray, pitch_data: dict) -> np.ndarray:
        h, w = tactical_canvas.shape[:2]
        pixel_points = {}
        
        # 1. Extract player positions and convert to pixel coordinates
        if pitch_data:
            for tid, pt in pitch_data.get('team_0', {}).items():
                px, py = self._real_to_pixel(pt[0], pt[1], w, h)
                pixel_points[(px, py)] = config.TEAM_0_COLOR
                
            for tid, pt in pitch_data.get('team_1', {}).items():
                px, py = self._real_to_pixel(pt[0], pt[1], w, h)
                pixel_points[(px, py)] = config.TEAM_1_COLOR

        # 2. Check if we have enough points to create a Voronoi diagram (at least 4 for meaningful control zones)
        if len(pixel_points) >= 4:
            self.last_pixel_points = pixel_points 
        else:
            pixel_points = self.last_pixel_points

        # 3. RENDER
        if len(pixel_points) < 4:
            return tactical_canvas

        controlled_canvas = self.engine.generate_overlay(tactical_canvas, pixel_points)
        return controlled_canvas