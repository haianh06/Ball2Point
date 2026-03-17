from ultralytics import YOLO
import numpy as np
import cv2
import torch
from .config import config

class SpatialEngine:
    """Xử lý ngầm phép chiếu từ Camera Pixel sang Tọa độ Mét trên sân"""
    def __init__(self):
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(config.KEYPOINT_MODEL_PATH)
        self.target_points = config.get_standard_pitch_points()
        
        self.matrix = None
        self.last_valid_matrix = None 

    def update_homography(self, frame: np.ndarray):
        results = self.model(frame, device=self.device, verbose=False)[0]
        
        # Kiểm tra nếu model không bắt được bất kỳ điểm nào
        has_keypoints = hasattr(results, 'keypoints') and results.keypoints is not None and len(results.keypoints.data) > 0
        if not has_keypoints:
            self.matrix = self.last_valid_matrix
            return

        kpts = results.keypoints.data.cpu().numpy()
        frame_kpts = kpts[0]
        
        if len(frame_kpts) != 29:
            self.matrix = self.last_valid_matrix
            return

        src_pts, dst_pts = [], []
        for i in range(29):
            x, y, conf = frame_kpts[i]
            if conf > config.CONFIDENCE_THRESHOLD:
                src_pts.append([x, y])
                dst_pts.append(self.target_points[i])

        if len(src_pts) >= 4:
            src_arr = np.array(src_pts, dtype=np.float32)
            dst_arr = np.array(dst_pts, dtype=np.float32)
            matrix, _ = cv2.findHomography(src_arr, dst_arr, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            
            if matrix is not None:
                self.matrix = matrix
                self.last_valid_matrix = matrix  # Cập nhật cache
            else:
                self.matrix = self.last_valid_matrix
        else:
            self.matrix = self.last_valid_matrix

    def pixels_to_meters(self, boxes: list) -> list:
        if not boxes:
            return []

        # Lấy tâm đáy (Gót chân cầu thủ)
        bottom_centers = np.array([
            [(box[0] + box[2]) / 2.0, box[3]]
            for box in boxes
        ], dtype=np.float32)

        # TRƯỜNG HỢP 1: Có ma trận (từ frame hiện tại hoặc lấy từ Cache)
        if self.matrix is not None:
            bottom_centers_reshaped = bottom_centers.reshape(-1, 1, 2)
            try:
                pitch_points = cv2.perspectiveTransform(bottom_centers_reshaped, self.matrix)
                return pitch_points.reshape(-1, 2).tolist()
            except Exception:
                pass

        # TRƯỜNG HỢP 2: Fallback (Ước lượng thô từ pixel sang mét)
        # Hệ thống vẫn tiếp tục tính được vận tốc thay vì ném ra mảng rỗng
        fallback_points = bottom_centers * config.PIXEL_TO_METER_FALLBACK
        return fallback_points.tolist()