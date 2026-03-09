import cv2
import numpy as np
from config import config
from keypoint_detector import KeypointDetector
from homography import HomographyEngine
from pitch_renderer import StaticPitchRenderer

class TacticalPipeline:
    def __init__(self):
        self.detector = KeypointDetector(config.KEYPOINT_MODEL_PATH, config.CONFIDENCE_THRESHOLD)
        self.homography = HomographyEngine()
        self.renderer = StaticPitchRenderer()
        self.last_pitch_data = None 

    def process_frame(self, frame: np.ndarray, frame_idx: int, tracking_data: dict) -> tuple:
        kpts = self.detector.detect(frame)
        self.homography.update_matrix(kpts)
        
        players_dict = tracking_data.get('player', {}).get(frame_idx, {})
        team_ids_dict = tracking_data.get('player_team_ids', {}).get(frame_idx, {})
        goalkeeper_boxes_dict = tracking_data.get('goalkeeper', {}).get(frame_idx, {})
        referee_boxes_dict = tracking_data.get('referee', {}).get(frame_idx, {})
        ball_box = tracking_data.get('ball', {}).get(frame_idx)
        
        pitch_data_with_ids = {'team_0': {}, 'team_1': {}, 'goalkeepers': {}, 'referees': {}, 'ball': None}

        if self.homography.transformer is not None:
            for tid, box in players_dict.items():
                # CHỐT CHẶN BẢO VỆ: Bỏ qua metadata tốc độ bị module khác nhét vào
                if tid == 'speed_info':
                    continue
                
                pitch_pt = self.homography.transform_boxes_to_pitch([box])
                if pitch_pt:
                    pt = pitch_pt[0]
                    team = team_ids_dict.get(tid, 0)
                    pitch_data_with_ids[f'team_{team}'][tid] = pt
            
            for tid, box in referee_boxes_dict.items():
                pitch_pt = self.homography.transform_boxes_to_pitch([box])
                if pitch_pt: pitch_data_with_ids['referees'][tid] = pitch_pt[0]
                
            for tid, box in goalkeeper_boxes_dict.items():
                pitch_pt = self.homography.transform_boxes_to_pitch([box])
                if pitch_pt: pitch_data_with_ids['goalkeepers'][tid] = pitch_pt[0]
                
            if ball_box is not None:
                ball_pitch = self.homography.transform_boxes_to_pitch([ball_box])
                if ball_pitch: pitch_data_with_ids['ball'] = ball_pitch[0]
                
            self.last_pitch_data = pitch_data_with_ids
            status_text = "Homography: Tracking"
            status_color = (0, 255, 0)
        else:
            pitch_data_with_ids = self.last_pitch_data if self.last_pitch_data else pitch_data_with_ids
            status_text = "Homography: Fallback"
            status_color = (0, 0, 255)

        tactical_view = self.renderer.render(pitch_data_with_ids)
        cv2.putText(tactical_view, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        return tactical_view, pitch_data_with_ids