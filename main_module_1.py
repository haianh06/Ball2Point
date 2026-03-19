import cv2
import os
from tqdm import tqdm
from module_1.pipeline import Module1Pipeline
from module_1.annotator import SoccerAnnotator
from module_1.config import ModelConfig
from module_1.post_processor import PostProcessor

def main():
    if not os.path.exists(ModelConfig.INPUT_VIDEO_PATH):
        print(f"Cannot find file at {ModelConfig.INPUT_VIDEO_PATH}")
        return

    print("=== PHASE 1: Detection & Tracking ===")
    pipeline = Module1Pipeline()
    max_frames = 500
    tracking_results = pipeline.process_video(
        video_path=ModelConfig.INPUT_VIDEO_PATH, 
        max_frames=max_frames
    )

    print("\n=== PHASE 2: INTERPOLATION ===")
    tracking_data = PostProcessor.interpolate_ball(tracking_data, max_gap=30)
    
    print("\n=== PHASE 3: RENDER VIDEO ===")
    annotator = SoccerAnnotator()
    cap = cv2.VideoCapture(ModelConfig.INPUT_VIDEO_PATH)

    os.makedirs(os.path.dirname(ModelConfig.OUTPUT_VIDEO_PATH), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        ModelConfig.OUTPUT_VIDEO_PATH, 
        fourcc, 
        25.0, 
        (int(cap.get(3)), int(cap.get(4)))
    )

    print("\n=== PHASE 3: RENDER VIDEO (DISK I/O TRADEOFF) ===")
    annotator = SoccerAnnotator()
    cap = cv2.VideoCapture(ModelConfig.INPUT_VIDEO_PATH)
    
    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ModelConfig.OUTPUT_VIDEO_PATH, fourcc, 30.0, 
                         (int(cap.get(3)), int(cap.get(4))))

    for idx in tqdm(range(max_frames), desc="Rendering frames"):
        ret, frame = cap.read()
        if not ret: break
        
        annotated_frame = annotator.draw(frame, idx, tracking_results)
        
        out.write(annotated_frame)

    cap.release()
    out.release()
    print("COMPLETE!")

if __name__ == "__main__":
    main()