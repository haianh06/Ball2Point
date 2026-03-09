import pandas as pd

class PostProcessor:
    @staticmethod
    def interpolate_ball(tracking_data: dict, max_gap: int = 30) -> dict:
        """Nội suy tuyến tính để điền vào các frame bị mất dấu bóng."""
        ball_data = tracking_data.get('ball', {})
        if not ball_data:
            return tracking_data
        
        # Chuyển đổi sang DataFrame để tận dụng hàm interpolate của pandas
        df = pd.DataFrame.from_dict(ball_data, orient='index')
        if df.empty:
            return tracking_data
            
        df.columns = ['x1', 'y1', 'x2', 'y2']
        
        # Tạo index liên tục từ frame đầu đến frame cuối bóng xuất hiện
        min_idx, max_idx = df.index.min(), df.index.max()
        df = df.reindex(range(min_idx, max_idx + 1))
        
        # Nội suy tuyến tính, giới hạn số frame liên tiếp tối đa được nội suy (max_gap)
        df = df.interpolate(method='linear', limit=max_gap, limit_direction='both')
        
        # Chuyển ngược lại dạng dict
        tracking_data['ball'] = {i: row.tolist() for i, row in df.dropna().iterrows()}
        return tracking_data