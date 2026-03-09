import os
import cv2
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from module_1.pipeline import Module1Pipeline
from module_1.post_processor import PostProcessor
from module_1.annotator import SoccerAnnotator
from module_1.io_utils import get_video_generator, get_video_info

from module_2.pipeline import TacticalPipeline
from pipeline import PitchControlPipeline

def resize_to_match_height(img1: np.ndarray, img2: np.ndarray):
    h1 = img1.shape[0]
    h2, w2 = img2.shape[:2]
    if h1 != h2:
        new_w2 = int(w2 * (h1 / h2))
        img2 = cv2.resize(img2, (new_w2, h1))
    return img1, img2

def main():
    VIDEO_PATH = r"inputs/test_1.mp4"
    OUTPUT_PATH = r"outputs/voronoi_analysis.mp4"
    MAX_FRAMES = -1
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    video_info = get_video_info(VIDEO_PATH)

    print("[1/4] Booting Models...")
    mod1_pipeline = Module1Pipeline()
    mod2_pipeline = TacticalPipeline()
    mod3_pipeline = PitchControlPipeline()
    annotator = SoccerAnnotator()

    print("\n[2/4] PHASE 1: Detection, Tracking & Clustering...")
    tracking_data = mod1_pipeline.process_video(VIDEO_PATH, max_frames=MAX_FRAMES)
    tracking_data = PostProcessor.interpolate_ball(tracking_data, max_gap=30)

    print("\n[3/4] PHASE 2: Rendering Tactical Map & Voronoi Space Control...")
    total_frames = video_info.total_frames if MAX_FRAMES == -1 else min(MAX_FRAMES, video_info.total_frames)
    generator = get_video_generator(VIDEO_PATH, max_frames=MAX_FRAMES)

    out = None

    for frame_idx, frame in tqdm(enumerate(generator), total=total_frames):
        # 1. Vẽ Camera gốc (Chỉ chứa ID và Team)
        annotated_frame = annotator.draw(frame.copy(), frame_idx, tracking_data)
        
        # 2. Xử lý Module 2: Trích xuất tọa độ mét và render bản đồ 2D cơ bản
        tactical_frame, pitch_data_with_ids = mod2_pipeline.process_frame(frame, frame_idx, tracking_data)
        
        # 3. Xử lý Module 3: Phủ lớp Voronoi lên bản đồ 2D
        # Lưu ý: mod2_pipeline.last_pitch_data chứa dữ liệu mét đã được bóc tách
        pitch_data = mod2_pipeline.last_pitch_data if mod2_pipeline.last_pitch_data else {}
        tactical_frame = mod3_pipeline.process_frame(tactical_frame, pitch_data)
        
        # 4. Ghép ảnh Side-by-side
        annotated_frame, tactical_frame = resize_to_match_height(annotated_frame, tactical_frame)
        combined_frame = np.hstack((annotated_frame, tactical_frame))
        
        if out is None:
            h, w = combined_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(OUTPUT_PATH, fourcc, video_info.fps, (w, h))
        out.write(combined_frame)

    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\n[DONE] Pipeline hoàn tất. Video đã lưu tại: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()