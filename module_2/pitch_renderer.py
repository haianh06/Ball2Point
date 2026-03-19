import cv2
import numpy as np
from .config import config

class StaticPitchRenderer:
    def __init__(self):
        self.base_pitch = cv2.imread(config.PITCH_IMAGE_PATH)
        if self.base_pitch is None:
            raise FileNotFoundError(f"Can not find pitch image: {config.PITCH_IMAGE_PATH}")
            
        self.img_height, self.img_width = self.base_pitch.shape[:2]
        
        # Calculate margins and drawable area based on the pitch image dimensions
        self.margin_x = int(self.img_width * 0.046)
        self.margin_y = int(self.img_height * 0.046)
        self.draw_width = self.img_width - 2 * self.margin_x
        self.draw_height = self.img_height - 2 * self.margin_y
        
        self.team_colors = {
            1: (255, 0, 0),    # Blue (Team 1)
            0: (0, 0, 255)     # Red (Team 0)
        }
        self.gk_color = (0, 255, 255)    # Yellow (Goalkeeper)
        self.ref_color = (0, 0, 0)       # Black (Referee)
        self.ball_color = (255, 255, 255)# White (Ball)

    def _real_to_pixel(self, x_real: float, y_real: float) -> tuple:
        x_pct = x_real / config.REAL_LENGTH
        y_pct = y_real / config.REAL_WIDTH
        x_px = self.margin_x + int(x_pct * self.draw_width)
        y_px = self.margin_y + int(y_pct * self.draw_height)
        return max(0, min(self.img_width - 1, x_px)), max(0, min(self.img_height - 1, y_px))

    def render(self, pitch_data_with_ids: dict, speed_data: dict = None) -> np.ndarray:
        canvas = self.base_pitch.copy()
        if speed_data is None: speed_data = {}
        
        # Draw players with ID 1 and 0, using team colors and speed-based halo rings
        for team_key, default_color in [('team_0', self.team_colors[0]), ('team_1', self.team_colors[1])]:
            for tid, pt in pitch_data_with_ids.get(team_key, {}).items():
                x, y = self._real_to_pixel(pt[0], pt[1])
                
                # Extract speed info for this player if available to determine halo color
                speed_info = speed_data.get(tid)
                ring_color = speed_info['color'] if speed_info else (255, 255, 255)
                
                # Draw halo ring
                cv2.circle(canvas, (x, y), 11, ring_color, -1)
                # Draw color-filled circle for player
                cv2.circle(canvas, (x, y), 7, default_color, -1)
                
                # Present speed info as text if available and above threshold
                if speed_info and speed_info['speed'] > 6.0:
                    cv2.putText(canvas, f"{speed_info['speed']}m/s", (x+13, y+4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw Goalkeepers
        for tid, pt in pitch_data_with_ids.get('goalkeepers', {}).items():
            x, y = self._real_to_pixel(pt[0], pt[1])
            cv2.circle(canvas, (x, y), 9, self.gk_color, -1)
            cv2.circle(canvas, (x, y), 9, (0, 0, 0), 1)
            
        # Draw Referees
        for tid, pt in pitch_data_with_ids.get('referees', {}).items():
            x, y = self._real_to_pixel(pt[0], pt[1])
            cv2.rectangle(canvas, (x-6, y-6), (x+6, y+6), self.ref_color, -1)

        # Draw Ball
        ball_pt = pitch_data_with_ids.get('ball')
        if ball_pt is not None:
            x, y = self._real_to_pixel(ball_pt[0], ball_pt[1])
            cv2.circle(canvas, (x, y), 6, self.ball_color, -1)
            cv2.circle(canvas, (x, y), 6, (0, 0, 0), 2)
            
        return canvas