import os
import cv2
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
from module_4.pipeline import HeatmapPipeline

def main():
    VIDEO_PATH = r"inputs/test_1.mp4"
    OUTPUT_PATH = r"outputs/tracking_video.mp4"
    MAX_FRAMES = -1
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    video_info = get_video_info(VIDEO_PATH)

    print("[1/4] Booting Models...")
    mod1_pipeline = Module1Pipeline()
    mod2_pipeline = TacticalPipeline()
    mod4_pipeline = HeatmapPipeline() # Khởi tạo Module 4
    annotator = SoccerAnnotator()

    print("\n[2/4] PHASE 1: Tracking...")
    tracking_data = mod1_pipeline.process_video(VIDEO_PATH, max_frames=MAX_FRAMES)
    tracking_data = PostProcessor.interpolate_ball(tracking_data, max_gap=30)

    print("\n[3/4] PHASE 2: Video Rendering & Heatmap Accumulation...")
    total_frames = video_info.total_frames if MAX_FRAMES == -1 else min(MAX_FRAMES, video_info.total_frames)
    generator = get_video_generator(VIDEO_PATH, max_frames=MAX_FRAMES)

    out = None

    for frame_idx, frame in tqdm(enumerate(generator), total=total_frames):
        # 1. Trích xuất Tọa độ từ Module 2
        tactical_frame, pitch_data_with_ids = mod2_pipeline.process_frame(frame, frame_idx, tracking_data)
        
        # 2. MODULE 4: Tích lũy tọa độ ngầm (Không vẽ lên video)
        pitch_data = mod2_pipeline.last_pitch_data if mod2_pipeline.last_pitch_data else {}
        mod4_pipeline.process_frame(pitch_data)
        
        # 3. Vẽ Video Gốc (Chỉ vẽ ID cầu thủ, nếu không muốn ghép Tactical_frame thì bỏ qua)
        annotated_frame = annotator.draw(frame.copy(), frame_idx, tracking_data)
        
        # (Chỗ này cậu có thể hstack với tactical_frame hoặc chỉ xuất ảnh gốc tùy ý)
        if out is None:
            h, w = annotated_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(OUTPUT_PATH, fourcc, video_info.fps, (w, h))
        out.write(annotated_frame)

    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    print("\n[4/4] PHASE 3: Exporting Heatmap Images...")
    # XUẤT ẢNH: Lệnh này sẽ bung toàn bộ data đã tích lũy thành Folder Ảnh
    mod4_pipeline.export_results()
    
    print(f"\n[DONE] Pipeline hoàn tất.")

if __name__ == "__main__":
    main()