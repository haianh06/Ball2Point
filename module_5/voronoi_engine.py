import cv2
import numpy as np
from config import config

class VoronoiEngine:
    def __init__(self):
        self.alpha = config.ALPHA_BLEND

    def generate_overlay(self, canvas: np.ndarray, pixel_points_dict: dict) -> np.ndarray:
        """
        Tạo lớp phủ đa giác Voronoi lên trên 2D Map.
        pixel_points_dict format: { (x_pixel, y_pixel): color_tuple }
        """
        h, w = canvas.shape[:2]
        
        # Subdiv2D yêu cầu bound (kích thước giới hạn của khung vẽ)
        subdiv = cv2.Subdiv2D((0, 0, w, h))
        
        valid_points = {}
        for (x, y), color in pixel_points_dict.items():
            # Clamp: Ép tọa độ không được vượt ra ngoài mép ảnh (sẽ gây crash C++)
            x = max(1, min(w - 2, int(x)))
            y = max(1, min(h - 2, int(y)))
            
            # Khử trùng lặp: Subdiv2D sẽ lỗi nếu có 2 điểm trùng tọa độ chính xác
            if (x, y) not in valid_points:
                valid_points[(x, y)] = color
                subdiv.insert((x, y))

        # Thuật toán yêu cầu ít nhất 4 điểm để tạo lưới
        if len(valid_points) < 4:
            return canvas

        facets, centers = subdiv.getVoronoiFacetList([])
        overlay = np.zeros_like(canvas, dtype=np.uint8)

        for facet, center in zip(facets, centers):
            cx, cy = center
            min_dist = float('inf')
            facet_color = None
            
            # Thuật toán tìm hàng xóm gần nhất (Nearest Neighbor) để gán màu
            for (vx, vy), color in valid_points.items():
                dist = (vx - cx)**2 + (vy - cy)**2
                if dist < min_dist:
                    min_dist = dist
                    facet_color = color

            if facet_color:
                ifacet = np.array([facet], np.int32)
                # Đổ màu đặc vào đa giác
                cv2.fillConvexPoly(overlay, ifacet[0], facet_color)
                if config.DRAW_BORDERS:
                    # Vẽ viền ranh giới
                    cv2.polylines(overlay, [ifacet[0]], True, config.BORDER_COLOR, 1)

        # Alpha Blending: Chỉ áp dụng độ trong suốt lên những pixel có chứa mảng màu Voronoi
        mask = np.any(overlay != 0, axis=-1)
        result = canvas.copy()
        result[mask] = cv2.addWeighted(canvas[mask], 1 - self.alpha, overlay[mask], self.alpha, 0)
        
        return result