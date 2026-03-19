from .heatmap_engine import HeatmapEngine
from module_2.pitch_renderer import StaticPitchRenderer

class HeatmapPipeline:
    def __init__(self):
        self.scaler = StaticPitchRenderer()
        self.engine = HeatmapEngine(self.scaler.img_width, self.scaler.img_height)

    def process_frame(self, pitch_data: dict):
        if not pitch_data:
            return

        for team_key in ['team_0', 'team_1']:
            for tid, pt in pitch_data.get(team_key, {}).items():
                px, py = self.scaler._real_to_pixel(pt[0], pt[1])
                self.engine.update(tid, px, py)

    def export_results(self):
        self.engine.export_all_heatmaps()