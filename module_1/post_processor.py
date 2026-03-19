import pandas as pd

class PostProcessor:
    @staticmethod
    def interpolate_ball(tracking_data: dict, max_gap: int = 30) -> dict:
        """Iterpolate ball positions for frames where it was not detected, using linear interpolation."""
        ball_data = tracking_data.get('ball', {})
        if not ball_data:
            return tracking_data
        
        df = pd.DataFrame.from_dict(ball_data, orient='index')
        if df.empty:
            return tracking_data
            
        df.columns = ['x1', 'y1', 'x2', 'y2']
        
        min_idx, max_idx = df.index.min(), df.index.max()
        df = df.reindex(range(min_idx, max_idx + 1))
        
        df = df.interpolate(method='linear', limit=max_gap, limit_direction='both')
        
        tracking_data['ball'] = {i: row.tolist() for i, row in df.dropna().iterrows()}
        return tracking_data