import cv2
import supervision as sv

def get_video_generator(video_path: str, max_frames: int = -1):
    """Đọc video dùng Generator để tránh Out Of Memory (OOM)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không thể mở video: {video_path}")
    
    count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success or (0 < max_frames <= count):
            break
        yield frame
        count += 1
        
    cap.release()

def get_video_info(video_path: str) -> sv.VideoInfo:
    return sv.VideoInfo.from_video_path(video_path)