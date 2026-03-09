import supervision as sv

class PlayerTracker:
    def __init__(self, track_thresh, track_buffer):
        self.tracker = sv.ByteTrack(
        track_activation_threshold=track_thresh, 
        lost_track_buffer=track_buffer
    )
        
    def update(self, detections: sv.Detections) -> sv.Detections:
        """Cập nhật ID cho detections cầu thủ."""
        if len(detections.xyxy) == 0:
            return detections
        return self.tracker.update_with_detections(detections)