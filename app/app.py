import os
import cv2
import json
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from module_1.pipeline import Module1Pipeline
from module_1.post_processor import PostProcessor
from module_1.annotator import SoccerAnnotator
from module_1.io_utils import get_video_generator, get_video_info

from module_2.pipeline import TacticalPipeline
from module_3.spatial_math import SpatialEngine
from module_3.speed_estimator import SpeedAndDistanceEstimator
from module_4.pipeline import HeatmapPipeline
from module_5.pipeline import PitchControlPipeline

# UI Config 
st.set_page_config(page_title="Ball2Point", page_icon="⚽", layout="wide")
PROJECT_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == 'app' else Path(__file__).resolve().parent

def resize_to_match_height(img1: np.ndarray, img2: np.ndarray):
    h1 = img1.shape[0]
    h2, w2 = img2.shape[:2]
    if h1 != h2:
        new_w2 = int(w2 * (h1 / h2))
        img2 = cv2.resize(img2, (new_w2, h1))
    return img1, img2

def convert_to_h264(input_path: str, output_path: str):
    """FFmpeg command to convert video to H.264 format for better web compatibility"""
    command = f'ffmpeg -y -i "{input_path}" -vcodec libx264 -crf 23 -preset fast "{output_path}"'
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def main():
    st.title("⚽ Ball2Point")
    st.markdown("A system trying to do anything with soccer videos")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Input & Settings")
        uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        
        st.markdown("---")
        st.subheader("Features Options")
        run_minimap = st.toggle("🗺️ Tactical Minimap", value=True)
        run_speed = st.toggle("⚡ Speed & Distance", value=True)
        run_heatmap = st.toggle("🔥 Physical Heatmap", value=True)
        run_voronoi = st.toggle("🛡️ Pitch Control (Voronoi)", value=True)
        
        st.markdown("---")
        st.subheader("Render Options")
        process_mode = st.radio("Processing Mode:", ["Full Video", "Test Only (500 Frames)"])
        
        start_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)

    # Main Logic
    if start_btn and uploaded_file is not None:
        input_path = str(PROJECT_ROOT / "inputs/temp_upload.mp4")
        temp_cv2_output = str(PROJECT_ROOT / "outputs/temp_render.mp4")
        final_web_output = str(PROJECT_ROOT / "outputs/final_dashboard.mp4")
        
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        os.makedirs(os.path.dirname(final_web_output), exist_ok=True)
        
        # Chungking
        with open(input_path, "wb") as f:
            bytes_data = uploaded_file.getbuffer()
            f.write(bytes_data)
                
        video_info = get_video_info(input_path)
        actual_frames = video_info.total_frames if process_mode == "Full Video" else min(200, video_info.total_frames)

        # UI Progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Models
        status_text.text("AI models is ready...")
        mod1 = Module1Pipeline()
        annotator = SoccerAnnotator()
        
        mod2 = TacticalPipeline() if (run_minimap or run_voronoi or run_heatmap) else None
        spatial_engine = SpatialEngine() if run_speed else None
        speed_estimator = SpeedAndDistanceEstimator(frame_rate=video_info.fps) if run_speed else None
        mod4 = HeatmapPipeline() if run_heatmap else None
        mod5 = PitchControlPipeline() if run_voronoi else None

        # PHASE 1
        status_text.text("[Phase 1/4] Detection & Tracking...")
        tracking_data = mod1.process_video(input_path, max_frames=actual_frames)
        tracking_data = PostProcessor.interpolate_ball(tracking_data, max_gap=30)
        progress_bar.progress(30)

        # PHASE 2
        if run_speed:
            status_text.text("[Phase 2/4] Spatial Transformation & Speed Logic...")
            spatial_data = {}
            gen = get_video_generator(input_path, max_frames=actual_frames)
            for frame_idx, frame in enumerate(gen):
                spatial_engine.update_homography(frame)
                players_dict = tracking_data.get('player', {}).get(frame_idx, {})
                spatial_data[frame_idx] = {}
                
                for tid, box in players_dict.items():
                    if tid != 'speed_info':
                        pts = spatial_engine.pixels_to_meters([box])
                        if pts: spatial_data[frame_idx][tid] = pts[0]
            
            tracking_data = speed_estimator.calculate_speed_and_distance(tracking_data, spatial_data)
        progress_bar.progress(50)

        # PHASE 3: LIVE PREVIEW & RENDER
        status_text.text(f"[Phase 3/4] Rendering Video ({actual_frames} frames)...")
        gen = get_video_generator(input_path, max_frames=actual_frames)
        out = None
        
        # Preview container
        preview_container = st.empty()

        for frame_idx, frame in enumerate(gen):
            ann_frame = annotator.draw(frame.copy(), frame_idx, tracking_data)
            final_frame = ann_frame

            if mod2 is not None:
                tac_frame, pitch_data = mod2.process_frame(frame, frame_idx, tracking_data)
                
                if run_voronoi:
                    pitch_data_fallback = mod2.last_pitch_data if mod2.last_pitch_data else {}
                    tac_frame = mod5.process_frame(tac_frame, pitch_data_fallback)

                if run_heatmap:
                    pitch_data_fallback = mod2.last_pitch_data if mod2.last_pitch_data else {}
                    mod4.process_frame(pitch_data_fallback)

                if run_minimap or run_voronoi:
                    ann_frame, tac_frame = resize_to_match_height(ann_frame, tac_frame)
                    final_frame = np.hstack((ann_frame, tac_frame))

            # --- LIVE PREVIEW ---
            if frame_idx % 5 == 0:
                progress_bar.progress(50 + int((frame_idx / actual_frames) * 30))
                preview_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                preview_container.image(
                    preview_rgb, 
                    caption=f"Rendering Progress: Frame {frame_idx}/{actual_frames} ⏳", 
                    use_container_width=True
                )

            if out is None:
                h, w = final_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_cv2_output, fourcc, video_info.fps, (w, h))
            out.write(final_frame)

        if out: out.release()
        cv2.destroyAllWindows()
        preview_container.empty()
        
        # PHASE 4
        status_text.text("[Phase 4/4] Encoding H.264 & Exporting Data...")
        progress_bar.progress(90)
        
        convert_to_h264(temp_cv2_output, final_web_output)
        
        if run_heatmap:
            mod4.export_results()
            
        progress_bar.progress(100)
        status_text.success("✅ Analysis completed!")

        # --- DASHBOARD ---
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Frames", value=f"{actual_frames}")
        with col2:
            st.metric(label="FPS Video", value=f"{video_info.fps}")
        with col3:
            st.metric(label="Resolution", value=f"{video_info.resolution_wh[0]}x{video_info.resolution_wh[1]}")

        tab1, tab2, tab3 = st.tabs(["🎥 Analysis Video", "📈 Physical Statistics", "🔥 Heatmap Library"])

        with tab1:
            if os.path.exists(final_web_output):
                st.video(final_web_output)
            else:
                st.error("Cannnot find the rendered video. Please check if the FFmpeg conversion was successful.")

        with tab2:
            stats_path = str(PROJECT_ROOT / "outputs/player_statistics.json")
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    stats_data = json.load(f)
                
                # JSON to Daraframe
                df = pd.DataFrame(stats_data).T
                df.index.name = "Player ID"
                
                # Average Speed Calculation
                total_time_sec = actual_frames / video_info.fps
                
                df["avg_speed_kmh"] = (df["total_distance_m"] / total_time_sec) * 3.6
                df["avg_speed_kmh"] = df["avg_speed_kmh"].round(2)
                
                df.rename(columns={
                    "total_distance_m": "Total Distance (m)",
                    "avg_speed_kmh": "Average Speed (m/s)"
                }, inplace=True)
                
                df_display = df[["Total Distance (m)", "Average Speed (m/s)"]]

                st.dataframe(
                    df_display.style.highlight_max(axis=0, color='#2e6b21'),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No physical statistics data available. Please enable the Speed Module.")

        with tab3:
            heatmap_dir = str(PROJECT_ROOT / "outputs/player_heatmaps")
            if os.path.exists(heatmap_dir) and run_heatmap:
                images = [os.path.join(heatmap_dir, f) for f in os.listdir(heatmap_dir) if f.endswith('.jpg')]
                if images:
                    cols = st.columns(3)
                    for idx, img_path in enumerate(images):
                        cols[idx % 3].image(img_path, caption=os.path.basename(img_path).split('.')[0], use_container_width=True)
                else:
                    st.info("No heatmaps found. The team may not have moved.")

    elif start_btn and uploaded_file is None:
        st.error("⚠️ A video file must be uploaded before starting the analysis.")

if __name__ == "__main__":
    main()