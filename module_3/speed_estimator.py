import numpy as np
import json
import os
from pathlib import Path

class SpeedAndDistanceEstimator:
    def __init__(self, frame_rate=30, frame_window=5):
        self.frame_rate = frame_rate
        self.frame_window = frame_window
        self.stats_output = str(Path(__file__).resolve().parent.parent / "outputs/player_statistics.json")

    def calculate_speed_and_distance(self, tracking_data: dict, spatial_data: dict) -> dict:
        total_distance = {}
        top_speeds = {}
        
        frame_indices = sorted(list(spatial_data.keys()))
        if not frame_indices:
            return tracking_data

        max_frame = frame_indices[-1]

        # Xử lý theo từng lô (Batch) 5 frame
        for frame_num in range(0, max_frame, self.frame_window):
            last_frame = min(frame_num + self.frame_window, max_frame)
            
            if frame_num not in spatial_data or last_frame not in spatial_data:
                continue
                
            start_data = spatial_data[frame_num]
            end_data = spatial_data[last_frame]

            for track_id, start_pos in start_data.items():
                if track_id not in end_data:
                    continue

                end_pos = end_data[track_id]
                
                # Tính khoảng cách Euclidean
                distance_covered = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
                time_elapsed = (last_frame - frame_num) / self.frame_rate
                
                if time_elapsed <= 0:
                    continue
                    
                speed_mps = distance_covered / time_elapsed
                speed_kmh = speed_mps * 3.6

                # CHỐT CHẶN VẬT LÝ: Nếu tốc độ > 40km/h (Gần bằng kỷ lục Usain Bolt)
                # Chắc chắn đó là do Jitter (nhiễu) của YOLO, ta ép về 0 để không làm hỏng dữ liệu thống kê
                if speed_kmh > 40.0:
                    speed_kmh = 0.0
                    distance_covered = 0.0

                # Cộng dồn
                if track_id not in total_distance:
                    total_distance[track_id] = 0.0
                    top_speeds[track_id] = 0.0
                
                total_distance[track_id] += distance_covered
                if speed_kmh > top_speeds[track_id]:
                    top_speeds[track_id] = speed_kmh

                # Fill ngược dữ liệu cho tất cả các frame trong cửa sổ
                for batch_frame in range(frame_num, last_frame + 1):
                    if batch_frame not in tracking_data.get('player', {}):
                        continue
                    if track_id not in tracking_data['player'][batch_frame]:
                        continue
                        
                    if 'speed_info' not in tracking_data['player'][batch_frame]:
                        tracking_data['player'][batch_frame]['speed_info'] = {}
                        
                    tracking_data['player'][batch_frame]['speed_info'][track_id] = {
                        'speed': round(speed_kmh, 1),
                        'distance': round(total_distance[track_id], 1)
                    }

        self._export_json(total_distance, top_speeds)
        return tracking_data

    def _export_json(self, distances: dict, top_speeds: dict):
        stats = {}
        for tid in distances.keys():
            stats[int(tid)] = {
                'total_distance_m': round(distances[tid], 2),
                'top_speed_kmh': round(top_speeds[tid], 2)
            }
        
        # Tự động tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(self.stats_output), exist_ok=True)
        with open(self.stats_output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)