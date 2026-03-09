import numpy as np
from .config import config
from .view import ViewTransformer

class HomographyEngine:
    def __init__(self):
        self.transformer = None
        # Lấy tọa độ chuẩn 29 điểm của FIFA từ config một lần duy nhất
        self.target_points_dict = config.get_standard_pitch_points()

    def update_matrix(self, frame_kpts: np.ndarray):
        """
        Cập nhật ma trận ViewTransformer chỉ cần nhận keypoints của frame (1 tham số)
        """
        if len(frame_kpts) != 29:
            self.transformer = None
            return

        src_pts, dst_pts = [], []
        for i in range(29):
            x, y, conf = frame_kpts[i]
            if conf > config.CONFIDENCE_THRESHOLD:
                src_pts.append([x, y])
                dst_pts.append(self.target_points_dict[i])

        if len(src_pts) >= 4:
            try:
                self.transformer = ViewTransformer(source=np.array(src_pts), target=np.array(dst_pts))
            except ValueError:
                self.transformer = None
        else:
            self.transformer = None

    def transform_boxes_to_pitch(self, boxes: list) -> list:
        if self.transformer is None or not boxes:
            return []

        # Fix lỗi Bottom-Center để chiếu chuẩn xác xuống mặt cỏ
        bottom_centers = np.array([
            [(box[0] + box[2]) / 2.0, box[3]]
            for box in boxes
        ])

        try:
            pitch_points = self.transformer.transform_points(bottom_centers)
            return pitch_points.tolist()
        except Exception:
            return []