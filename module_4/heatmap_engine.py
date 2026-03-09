import cv2
import numpy as np
import os
from .config import config

# Lấy ảnh gốc của sân cỏ từ Module 2 để làm nền
from module_2.pitch_renderer import StaticPitchRenderer

class HeatmapEngine:
    def __init__(self, img_width: int, img_height: int):
        self.width = img_width
        self.height = img_height
        # { tracker_id: numpy_array_float32 }
        self.accumulators = {} 
        self.renderer = StaticPitchRenderer()

    def update(self, tid: int, x: int, y: int):
        """Cộng dồn tần suất xuất hiện của cầu thủ TID tại tọa độ (x, y)"""
        if tid not in self.accumulators:
            self.accumulators[tid] = np.zeros((self.height, self.width), dtype=np.float32)
            
        if 0 <= x < self.width and 0 <= y < self.height:
            self.accumulators[tid][y, x] += 1.0

    def export_all_heatmaps(self):
        """Khâu Post-Processing: Render và lưu toàn bộ ảnh khi kết thúc video"""
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        base_pitch = self.renderer.base_pitch  # Nền sân cỏ tĩnh

        exported_count = 0
        for tid, accumulator in self.accumulators.items():
            max_val = np.max(accumulator)
            if max_val == 0:
                continue

            # 1. Lan tỏa nhiệt
            blurred = cv2.GaussianBlur(accumulator, config.GAUSSIAN_KERNEL, 0)

            # 2. Chuẩn hóa về 0-255
            max_blur = np.max(blurred)
            if max_blur > 0:
                normalized = (blurred / max_blur * 255).astype(np.uint8)
            else:
                normalized = blurred.astype(np.uint8)

            # 3. Phủ dải màu Colormap
            heatmap = cv2.applyColorMap(normalized, config.COLORMAP)

            # 4. Kỹ thuật Alpha Masking: Xóa phông nền những vùng không chạy qua
            alpha_channel = (normalized / 255.0) * config.ALPHA_BLEND
            alpha_channel = np.expand_dims(alpha_channel, axis=-1)

            # 5. Blend với mặt cỏ
            result = (heatmap * alpha_channel + base_pitch * (1 - alpha_channel)).astype(np.uint8)

            # 6. Viết Text ID lên góc ảnh để dễ nhận biết
            cv2.putText(result, f"Player ID: {tid}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            # 7. Ghi ra đĩa
            output_path = os.path.join(config.OUTPUT_DIR, f"heatmap_id_{tid}.jpg")
            cv2.imwrite(output_path, result)
            exported_count += 1
            
        print(f"\n[Module 4] Đã xuất thành công {exported_count} ảnh Heatmap tại: {config.OUTPUT_DIR}")