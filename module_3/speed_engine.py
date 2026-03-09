import numpy as np
from collections import deque
import json
from .config import config

class SpeedEngine:
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.window_size = config.WINDOW_SIZE
        # Hàng đợi lưu tọa độ các frame gần nhất để khử nhiễu: { tid: deque([(x,y), ...]) }
        self.history = {}
        
        # State lưu trữ vĩnh viễn để xuất báo cáo Streamlit
        self.statistics = {} 

    def update(self, players_meters_dict: dict) -> dict:
        """
        Input: { tracker_id: [x_meter, y_meter] }
        Output: { tracker_id: {'speed': float} }
        """
        results = {}
        current_ids = set(players_meters_dict.keys())

        # Dọn rác (Garbage Collection): Xóa history của ID đã biến mất khỏi camera để nhẹ RAM
        for tid in list(self.history.keys()):
            if tid not in current_ids:
                del self.history[tid]

        for tid, pt in players_meters_dict.items():
            if tid not in self.history:
                self.history[tid] = deque(maxlen=self.window_size)
            if tid not in self.statistics:
                # Khởi tạo profile cho cầu thủ mới
                self.statistics[tid] = {'total_distance_m': 0.0, 'speeds': [], 'top_speed_kmh': 0.0}

            self.history[tid].append(np.array(pt))
            
            # CỘNG DỒN QUÃNG ĐƯỜNG: Tính khoảng cách frame-by-frame
            if len(self.history[tid]) >= 2:
                step_dist = np.linalg.norm(self.history[tid][-1] - self.history[tid][-2])
                self.statistics[tid]['total_distance_m'] += step_dist

            # TÍNH TỐC ĐỘ HIỂN THỊ: Dùng sliding window để khử giật lag do YOLO
            if len(self.history[tid]) == self.window_size:
                dist_window = np.linalg.norm(self.history[tid][-1] - self.history[tid][0])
                time_seconds = self.window_size / self.fps
                speed_kmh = (dist_window / time_seconds) * 3.6
                
                # Lưu log
                self.statistics[tid]['speeds'].append(speed_kmh)
                if speed_kmh > self.statistics[tid]['top_speed_kmh']:
                    self.statistics[tid]['top_speed_kmh'] = speed_kmh
                    
                results[tid] = {'speed': round(speed_kmh, 1)}
            else:
                results[tid] = {'speed': 0.0}
                
        return results

    def export_statistics(self):
        """Xuất file JSON chứa khoảng cách & tốc độ trung bình cho Streamlit."""
        final_stats = {}
        for tid, data in self.statistics.items():
            speeds = data['speeds']
            avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
            
            final_stats[int(tid)] = {
                'total_distance_meters': round(data['total_distance_m'], 2),
                'average_speed_kmh': round(avg_speed, 2),
                'top_speed_kmh': round(data['top_speed_kmh'], 2)
            }
            
        with open(config.STATS_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=4)
        print(f"[Module 5] Đã xuất log thống kê cầu thủ ra: {config.STATS_OUTPUT_PATH}")