import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir

while not (PROJECT_ROOT / "module_1").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm
import supervision as sv
from .config import config
from .io_utils import get_video_generator, get_video_info
from .detector import SoccerDetector
from .tracker import PlayerTracker
from .team_assigner import TeamAssigner

class Module1Pipeline:
    def __init__(self):
        self.detector = SoccerDetector(config.DETECTION_MODEL_PATH)
        self.tracker = PlayerTracker(config.TRACKER_MATCH_THRESH, config.TRACKER_BUFFER)
        self.assigner = TeamAssigner(config.SIGLIP_MODEL_NAME, config.N_TEAMS, config.UMAP_COMPONENTS)
        
    def train_clustering(self, video_path: str, max_frames: int = 300, stride: int = 10):
        """Thu thập ngẫu nhiên các crop của cầu thủ để train model chia đội."""
        print("Đang thu thập dữ liệu train chia đội...")
        generator = get_video_generator(video_path, max_frames=max_frames)
        crops = []
        
        for idx, frame in enumerate(generator):
            if idx % stride != 0:
                continue
            players, _, _ = self.detector.detect(frame)
            if len(players.xyxy) > 0:
                crops.extend([sv.cv2_to_pillow(sv.crop_image(frame, box)) for box in players.xyxy])
                
        self.assigner.fit(crops)
        print("Huấn luyện chia đội hoàn tất!")

    def process_video(self, video_path: str, max_frames: int = -1) -> dict:
        """
        Chạy pipeline chính. Trả về dictionary tracking data.
        """
        if not self.assigner.is_trained:
            self.train_clustering(video_path)
            
        video_info = get_video_info(video_path)
        total_frames = video_info.total_frames if max_frames == -1 else min(max_frames, video_info.total_frames)
        generator = get_video_generator(video_path, max_frames)
        
        tracking_data = {
            'player': {}, 'ball': {}, 'referee': {}, 'player_team_ids': {}
        }
        
        print("Bắt đầu phân tích Detection & Tracking...")
        for frame_idx, frame in tqdm(enumerate(generator), total=total_frames):
            # 1. Detect
            players, ball, referees = self.detector.detect(frame)
            
            # 2. Track Players
            players = self.tracker.update(players)
            
            # 3. Assign Teams
            team_ids = self.assigner.predict(frame, players)
            
            # 4. Lưu data
            if len(players.xyxy) > 0:
                tracking_data['player'][frame_idx] = {
                    tid: box.tolist() for tid, box in zip(players.tracker_id, players.xyxy)
                }
                tracking_data['player_team_ids'][frame_idx] = {
                    tid: tid_team for tid, tid_team in zip(players.tracker_id, team_ids)
                }
            
            if len(ball.xyxy) > 0:
                tracking_data['ball'][frame_idx] = ball.xyxy[0].tolist()
                
            if len(referees.xyxy) > 0:
                tracking_data['referee'][frame_idx] = {
                    i: box.tolist() for i, box in enumerate(referees.xyxy)
                }
                
        return tracking_data