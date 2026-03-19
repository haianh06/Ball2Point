import numpy as np
import cv2
import supervision as sv

class SoccerAnnotator:
    def __init__(self):
        self.ellipse_annotator = sv.EllipseAnnotator(thickness=2)
        
        self.team_colors = {
            0: sv.Color(255, 0, 0), 
            1: sv.Color(0, 0, 255)
        }
        self.referee_color = sv.Color(0, 0, 0)
        self.ball_color = sv.Color(255, 255, 255)

    def draw(self, frame: np.ndarray, frame_idx: int, tracking_data: dict) -> np.ndarray:
        annotated = frame.copy()
        
        players_data = tracking_data.get('player', {}).get(frame_idx, {})
        team_data = tracking_data.get('player_team_ids', {}).get(frame_idx, {})
        speed_info_data = tracking_data.get('player', {}).get(frame_idx, {}).get('speed_info', {})
        
        # 1. Draw Players
        if players_data:
            boxes, tracker_ids, class_ids = [], [], []
            for tid, box in players_data.items():
                if tid != 'speed_info':
                    boxes.append(box)
                    tracker_ids.append(tid)
                    class_ids.append(team_data.get(tid, 0))
                
            detections = sv.Detections(
                xyxy=np.array(boxes),
                tracker_id=np.array(tracker_ids),
                class_id=np.array(class_ids)
            )
            
            for team_id, color in self.team_colors.items():
                team_dets = detections[detections.class_id == team_id]
                if len(team_dets) > 0:
                    self.ellipse_annotator.color = color
                    annotated = self.ellipse_annotator.annotate(annotated, team_dets)
                    
            for tid, box in zip(tracker_ids, boxes):
                info = speed_info_data.get(tid)
                if info:
                    speed = info['speed']
                    dist = info['distance']
                    if speed > 0:
                        x1, y1, x2, y2 = box
                        x_center = int((x1 + x2) / 2)
                        y_bottom = int(y2)
                        
                        text = f"{speed} m/s | {dist} m"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]

                        cv2.putText(annotated, text, 
                                    (x_center - text_size[0]//2, y_bottom + 10 + text_size[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # 2. Draw Referees
        refs_data = tracking_data.get('referee', {}).get(frame_idx, {})
        if refs_data:
            ref_boxes = list(refs_data.values())
            ref_dets = sv.Detections(xyxy=np.array(ref_boxes), class_id=np.zeros(len(ref_boxes), dtype=int))
            self.ellipse_annotator.color = self.referee_color
            self.ellipse_annotator.color_lookup = sv.ColorLookup.INDEX
            annotated = self.ellipse_annotator.annotate(annotated, ref_dets)

        # 3. Draw Ball
        ball_box = tracking_data.get('ball', {}).get(frame_idx)
        if ball_box is not None:
            ball_det = sv.Detections(xyxy=np.array([ball_box]), class_id=np.array([0]))
            triangle = sv.TriangleAnnotator(base=15, height=15)
            triangle.color = self.ball_color
            triangle.color_lookup = sv.ColorLookup.INDEX
            annotated = triangle.annotate(annotated, ball_det)

        return annotated