from __future__ import annotations
import cv2
import numpy as np

def _scale_point(x: float, y: float, scale_x: float, scale_y: float) -> tuple[int, int]:
    return int(round(x * scale_x)), int(round(y * scale_y))

def draw_pitch(
    config,
    frame_size: tuple[int, int] = (1050, 680),
    background_color: tuple[int, int, int] = (34, 139, 34),
    line_color: tuple[int, int, int] = (255, 255, 255),
    line_thickness: int = 2,
) -> np.ndarray:
    width, height = frame_size
    pitch = np.full((height, width, 3), background_color, dtype=np.uint8)
    sx = width / float(config.pitch_length)
    sy = height / float(config.pitch_width)

    def pt(x: float, y: float) -> tuple[int, int]:
        return _scale_point(x, y, sx, sy)

    cx = config.pitch_length / 2
    cy = config.pitch_width / 2
    pa_y1, pa_y2 = cy - config.penalty_area_width / 2, cy + config.penalty_area_width / 2
    ga_y1, ga_y2 = cy - config.goal_area_width / 2, cy + config.goal_area_width / 2
    left_pa_x, right_pa_x = config.penalty_area_depth, config.pitch_length - config.penalty_area_depth
    left_ga_x, right_ga_x = config.goal_area_depth, config.pitch_length - config.goal_area_depth

    cv2.rectangle(pitch, pt(0, 0), pt(config.pitch_length, config.pitch_width), line_color, line_thickness)
    cv2.line(pitch, pt(cx, 0), pt(cx, config.pitch_width), line_color, line_thickness)
    
    center_px = pt(cx, cy)
    radius_px = int(round(config.center_circle_radius * min(sx, sy)))
    cv2.circle(pitch, center_px, radius_px, line_color, line_thickness)
    cv2.circle(pitch, center_px, 4, line_color, -1)

    cv2.rectangle(pitch, pt(0, pa_y1), pt(left_pa_x, pa_y2), line_color, line_thickness)
    cv2.rectangle(pitch, pt(right_pa_x, pa_y1), pt(config.pitch_length, pa_y2), line_color, line_thickness)
    cv2.rectangle(pitch, pt(0, ga_y1), pt(left_ga_x, ga_y2), line_color, line_thickness)
    cv2.rectangle(pitch, pt(right_ga_x, ga_y1), pt(config.pitch_length, ga_y2), line_color, line_thickness)

    left_pm, right_pm = pt(config.penalty_mark_distance, cy), pt(config.pitch_length - config.penalty_mark_distance, cy)
    cv2.circle(pitch, left_pm, 4, line_color, -1)
    cv2.circle(pitch, right_pm, 4, line_color, -1)

    cv2.ellipse(pitch, left_pm, (radius_px, radius_px), 0, 310, 50, line_color, line_thickness)
    cv2.ellipse(pitch, right_pm, (radius_px, radius_px), 0, 130, 230, line_color, line_thickness)

    return pitch