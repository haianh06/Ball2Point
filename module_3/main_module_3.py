import os
import cv2
from tqdm import tqdm

from module_1.pipeline import Module1Pipeline
from module_1.post_processor import PostProcessor
from module_1.annotator import SoccerAnnotator
from module_1.io_utils import get_video_generator, get_video_info

from .spatial_math import SpatialEngine
from .speed_estimator import SpeedAndDistanceEstimator

def main():
    VIDEO_PATH = r"inputs/test_1.mp4"
    OUTPUT_PATH = r"outputs/speed_analysis_only.mp4"
    MAX_FRAMES = -1
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    video_info = get_video_info(VIDEO_PATH)

    print("[1/4] Khởi tạo hệ thống Tracking...")
    mod1_pipeline = Module1Pipeline()
    spatial_engine = SpatialEngine()
    # Setting giống Abdullah Tarek: Tính quãng đường mỗi 5 frames
    speed_estimator = SpeedAndDistanceEstimator(frame_rate=video_info.fps, frame_window=5)
    annotator = SoccerAnnotator()

    print("\n[2/4] PHASE 1: Tracking & Phân chia đội...")
    tracking_data = mod1_pipeline.process_video(VIDEO_PATH, max_frames=MAX_FRAMES)
    tracking_data = PostProcessor.interpolate_ball(tracking_data, max_gap=30)

    print("\n[3/4] PHASE 2: Chạy Spatial Math & Tính Vận tốc (Post-Processing)...")
    # Quét ngầm để lấy tọa độ mét
    generator = get_video_generator(VIDEO_PATH, max_frames=MAX_FRAMES)
    total_frames = video_info.total_frames if MAX_FRAMES == -1 else min(MAX_FRAMES, video_info.total_frames)
    
    spatial_data = {}
    for frame_idx, frame in tqdm(enumerate(generator), total=total_frames, desc="Tính toán không gian"):
        spatial_engine.update_homography(frame)
        players_dict = tracking_data.get('player', {}).get(frame_idx, {})
        
        spatial_data[frame_idx] = {}
        if spatial_engine.matrix is not None:
            for tid, box in players_dict.items():
                if tid != 'speed_info':
                    pts = spatial_engine.pixels_to_meters([box])
                    if pts:
                        spatial_data[frame_idx][tid] = pts[0]
                        
    # Tính vận tốc và ghi đè vào tracking_data giống hệt logic repo Tarek
    tracking_data = speed_estimator.calculate_speed_and_distance(tracking_data, spatial_data)

    print("\n[4/4] PHASE 3: Render Video...")
    generator = get_video_generator(VIDEO_PATH, max_frames=MAX_FRAMES)
    out = None

    for frame_idx, frame in tqdm(enumerate(generator), total=total_frames, desc="Vẽ Video"):
        # Vẽ trực tiếp lên frame camera, không có Minimap
        annotated_frame = annotator.draw(frame.copy(), frame_idx, tracking_data)
        
        if out is None:
            h, w = annotated_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(OUTPUT_PATH, fourcc, video_info.fps, (w, h))
        out.write(annotated_frame)

    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\n[DONE] Đã xuất video tại: {OUTPUT_PATH}")
    print("Dữ liệu đã được lưu tại: outputs/player_statistics.json")

if __name__ == "__main__":
    main()