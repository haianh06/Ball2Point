import cv2
import numpy as np
from .config import config

class StaticPitchRenderer:
    def __init__(self):
        self.base_pitch = cv2.imread(config.PITCH_IMAGE_PATH)
        if self.base_pitch is None:
            raise FileNotFoundError(f"LỖI: Không tìm thấy ảnh sân tại {config.PITCH_IMAGE_PATH}")
            
        self.img_height, self.img_width = self.base_pitch.shape[:2]
        
        # Bù trừ lề mép sân
        self.margin_x = int(self.img_width * 0.046)
        self.margin_y = int(self.img_height * 0.046)
        self.draw_width = self.img_width - 2 * self.margin_x
        self.draw_height = self.img_height - 2 * self.margin_y
        
        self.team_colors = {
            0: (255, 0, 0),    # Blue (Team 0)
            1: (0, 0, 255)     # Red (Team 1)
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
        
        # Vẽ Cầu thủ Team 0 và Team 1
        for team_key, default_color in [('team_0', self.team_colors[0]), ('team_1', self.team_colors[1])]:
            for tid, pt in pitch_data_with_ids.get(team_key, {}).items():
                x, y = self._real_to_pixel(pt[0], pt[1])
                
                # Trích xuất màu tốc độ (halo ring)
                speed_info = speed_data.get(tid)
                ring_color = speed_info['color'] if speed_info else (255, 255, 255)
                
                # Vẽ viền báo tốc độ (to hơn)
                cv2.circle(canvas, (x, y), 11, ring_color, -1)
                # Vẽ màu đội (nhỏ hơn nằm ở trong)
                cv2.circle(canvas, (x, y), 7, default_color, -1)
                
                # Hiển thị text km/h nếu tốc độ > đi bộ
                if speed_info and speed_info['speed'] > 6.0:
                    cv2.putText(canvas, f"{speed_info['speed']}km/h", (x+13, y+4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Vẽ Thủ môn
        for tid, pt in pitch_data_with_ids.get('goalkeepers', {}).items():
            x, y = self._real_to_pixel(pt[0], pt[1])
            cv2.circle(canvas, (x, y), 9, self.gk_color, -1)
            cv2.circle(canvas, (x, y), 9, (0, 0, 0), 1)
            
        # Vẽ Trọng tài
        for tid, pt in pitch_data_with_ids.get('referees', {}).items():
            x, y = self._real_to_pixel(pt[0], pt[1])
            cv2.rectangle(canvas, (x-6, y-6), (x+6, y+6), self.ref_color, -1)

        # Vẽ Bóng
        ball_pt = pitch_data_with_ids.get('ball')
        if ball_pt is not None:
            x, y = self._real_to_pixel(ball_pt[0], ball_pt[1])
            cv2.circle(canvas, (x, y), 6, self.ball_color, -1)
            cv2.circle(canvas, (x, y), 6, (0, 0, 0), 2)
            
        return canvas