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
        self.accumulators = {} 
        self.renderer = StaticPitchRenderer()

    def update(self, tid: int, x: int, y: int):
        """Accumulate the frequency of appearance of player TID at coordinates (x, y)"""
        if tid not in self.accumulators:
            self.accumulators[tid] = np.zeros((self.height, self.width), dtype=np.float32)
            
        if 0 <= x < self.width and 0 <= y < self.height:
            self.accumulators[tid][y, x] += 1.0

    def export_all_heatmaps(self):
        """Render and export heatmap images for all tracked players"""
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        base_pitch = self.renderer.base_pitch

        exported_count = 0
        for tid, accumulator in self.accumulators.items():
            max_val = np.max(accumulator)
            if max_val == 0:
                continue

            # 1. Heatmap smoothing with Gaussian Blur to create a more visually appealing heatmap
            blurred = cv2.GaussianBlur(accumulator, config.GAUSSIAN_KERNEL, 0)

            # 2. Normalize to 0-255
            max_blur = np.max(blurred)
            if max_blur > 0:
                normalized = (blurred / max_blur * 255).astype(np.uint8)
            else:
                normalized = blurred.astype(np.uint8)

            # 3. Colorize heatmap
            heatmap = cv2.applyColorMap(normalized, config.COLORMAP)

            # 4. Alpha Masking to create transparency effect (0.0 - only heatmap, 1.0 - only original)
            alpha_channel = (normalized / 255.0) * config.ALPHA_BLEND
            alpha_channel = np.expand_dims(alpha_channel, axis=-1)

            # 5. Blend heatmap with original pitch image
            result = (heatmap * alpha_channel + base_pitch * (1 - alpha_channel)).astype(np.uint8)

            # 6. ID at the top-left corner
            cv2.putText(result, f"Player ID: {tid}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            # 7. Save
            output_path = os.path.join(config.OUTPUT_DIR, f"heatmap_id_{tid}.jpg")
            cv2.imwrite(output_path, result)
            exported_count += 1
            
        print(f"\n[Module 4] Done {exported_count} heatmap images at: {config.OUTPUT_DIR}")