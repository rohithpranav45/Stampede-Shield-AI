import cv2
import pandas as pd
import numpy as np
import imageio
import argparse
from scipy.ndimage import gaussian_filter


def draw_heatmap(frame, points, sigma=20, accumulate_heatmap=None):
    h, w, _ = frame.shape
    heatmap = np.zeros((h, w), dtype=np.float32) if accumulate_heatmap is None else accumulate_heatmap.copy()
    for x, y in points:
        if 0 <= int(y) < h and 0 <= int(x) < w:
            heatmap[int(y), int(x)] += 1.0
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    return heatmap


def overlay_heatmap_on_frame(frame, heatmap, alpha=0.6):
    heatmap_norm = np.clip(heatmap / (np.max(heatmap) + 1e-6), 0, 1)
    heatmap_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def animate_heatmap_on_video(video_path, traj_file, output_file="heatmap_on_video.gif", delay=0.2, mode="accumulate", sigma=20):
    # Load trajectory data
    if traj_file.endswith('.parquet'):
        df = pd.read_parquet(traj_file)
    else:
        df = pd.read_csv(traj_file)
    # Open video
    cap = cv2.VideoCapture(video_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = []
    frame_idx = 0
    accumulate_heatmap = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_tracks = df[df['frame'] == frame_idx]
        points = frame_tracks[['x', 'y']].values if not frame_tracks.empty else []
        if mode == "accumulate":
            accumulate_heatmap = draw_heatmap(frame, points, sigma=sigma, accumulate_heatmap=accumulate_heatmap)
            heatmap = accumulate_heatmap
        else:  # instant
            heatmap = draw_heatmap(frame, points, sigma=sigma)
        overlay = overlay_heatmap_on_frame(frame, heatmap)
        frame_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_idx += 1
        print(f"Frame {frame_idx} done")
    cap.release()
    imageio.mimsave(output_file, frames, fps=1/delay, loop=0)
    print(f"Heatmap animation saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate Gaussian heatmap overlay on video frames.")
    parser.add_argument('--video', type=str, required=True, help='Input video file')
    parser.add_argument('--traj_file', type=str, required=True, help='Trajectory Parquet or CSV file (from DeepSORT logging)')
    parser.add_argument('--output', type=str, default='heatmap_on_video.gif', help='Output GIF filename')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay (seconds) between frames')
    parser.add_argument('--mode', type=str, choices=['accumulate', 'instant'], default='accumulate', help='Heatmap mode: accumulate or instant')
    parser.add_argument('--sigma', type=float, default=20, help='Gaussian sigma for heatmap')
    args = parser.parse_args()
    animate_heatmap_on_video(args.video, args.traj_file, output_file=args.output, delay=args.delay, mode=args.mode, sigma=args.sigma) 