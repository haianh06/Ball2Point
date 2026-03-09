import cv2
import os
from tqdm import tqdm
from .pipeline import Module1Pipeline
from .annotator import SoccerAnnotator
from .config import ModelConfig

def main():
    if not os.path.exists(ModelConfig.INPUT_VIDEO_PATH):
        print(f"LỖI: Không tìm thấy file tại {ModelConfig.INPUT_VIDEO_PATH}")
        return

    print("=== BẮT ĐẦU PHASE 1: PIPELINE PHÂN TÍCH ===")
    pipeline = Module1Pipeline()
    max_frames = 500
    # Sử dụng process_video thay vì process_frame
    # Hàm này đã bao gồm: train chia đội, detect, track và gán team
    tracking_results = pipeline.process_video(
        video_path=ModelConfig.INPUT_VIDEO_PATH, 
        max_frames=max_frames
    )

    print("\n=== BẮT ĐẦU PHASE 2: NỘI SUY QUỸ ĐẠO BÓNG ===")
    # (Giữ nguyên logic nội suy của bạn)

    print("\n=== BẮT ĐẦU PHASE 3: RENDER VIDEO ===")
    annotator = SoccerAnnotator()
    cap = cv2.VideoCapture(ModelConfig.INPUT_VIDEO_PATH)
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(os.path.dirname(ModelConfig.OUTPUT_VIDEO_PATH), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Đổi sang XVID cho file .avi
    out = cv2.VideoWriter(
        ModelConfig.OUTPUT_VIDEO_PATH, 
        fourcc, 
        25.0, 
        (int(cap.get(3)), int(cap.get(4)))
    )

    print("\n=== BẮT ĐẦU PHASE 3: RENDER VIDEO (DISK I/O TRADEOFF) ===")
    annotator = SoccerAnnotator()
    cap = cv2.VideoCapture(ModelConfig.INPUT_VIDEO_PATH)
    
    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ModelConfig.OUTPUT_VIDEO_PATH, fourcc, 30.0, 
                         (int(cap.get(3)), int(cap.get(4))))

    for idx in tqdm(range(max_frames), desc="Rendering frames"):
        ret, frame = cap.read()
        if not ret: break
        
        # Vẽ dựa trên data đã xử lý ở Phase 1
        annotated_frame = annotator.draw(frame, idx, tracking_results)
        
        out.write(annotated_frame)

    cap.release()
    out.release()
    print("HOÀN TẤT!")

if __name__ == "__main__":
    main()