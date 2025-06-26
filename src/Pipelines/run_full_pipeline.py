import os
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage import gaussian_filter

def get_color(id):
    np.random.seed(int(id))
    return np.random.rand(3,)

def run_detection_tracking(video_path, checkpoints_dir, traj_file, interval=0.2):
    # Assumes run_yolov8x.py exists and outputs the trajectory file
    cmd = [
        "python", "run_yolov8x.py",
        "--video", video_path,
        "--interval", str(interval),
        "--checkpoints_dir", checkpoints_dir
    ]
    print("Running detection and tracking...")
    subprocess.run(cmd, check=True)
    # Find the latest parquet or csv in checkpoints_dir if traj_file not created directly
    if not os.path.exists(traj_file):
        files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.parquet') or f.endswith('.csv')]
        if files:
            files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
            traj_file_found = os.path.join(checkpoints_dir, files[0])
            print(f"Using {traj_file_found} as trajectory file.")
            return traj_file_found
        else:
            raise FileNotFoundError(f"No trajectory file found in {checkpoints_dir}.")
    return traj_file

def add_time_to_trajectories(traj_file, interval, output_file):
    if traj_file.endswith('.parquet'):
        df = pd.read_parquet(traj_file)
    else:
        df = pd.read_csv(traj_file)
    # Try to find the frame column
    frame_col = None
    for col in ['frame', 'frame_id', 'frame_number', 'frameIndex', 'image_id', 'frame_idx']:
        if col in df.columns:
            frame_col = col
            break
    if frame_col is None:
        raise ValueError("Trajectory file must have a 'frame' column or a known alternative.")
    df['time'] = df[frame_col] * interval
    df.to_parquet(output_file)
    print(f"Saved trajectories with time to {output_file}")
    return output_file

def run_gaussian_topview(traj_file, output_gif="gaussian_topview.gif", delay=0.2, sigma=10):
    # Load trajectory data
    df = pd.read_parquet(traj_file)
    ids = df['id'].unique()
    id_colors = {id_: get_color(id_) for id_ in ids}
    frame_nums = sorted(df['frame'].unique())
    all_x, all_y = df['x'].values, df['y'].values
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    frames = []

    for f in frame_nums:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('black')
        # Draw locus lines and points
        for id_ in ids:
            track = df[(df['id'] == id_) & (df['frame'] <= f)]
            if len(track) == 0:
                continue
            color = id_colors[id_]
            ax.plot(track['x'], track['y'], color=color, linewidth=2)
            ax.scatter(track['x'].iloc[-1], track['y'].iloc[-1], color=color, s=40, edgecolor='white', zorder=10)
        # Draw Gaussian heatmap for current points
        current_points = df[df['frame'] == f][['x', 'y']].values
        if len(current_points) > 0:
            heatmap, xedges, yedges = np.histogram2d(
                current_points[:, 0], current_points[:, 1],
                bins=200, range=[[x_min, x_max], [y_min, y_max]]
            )
            heatmap = gaussian_filter(heatmap, sigma=sigma)
            extent = [x_min, x_max, y_min, y_max]
            ax.imshow(
                heatmap.T, extent=extent, origin='lower', cmap='jet', alpha=0.5, aspect='auto', zorder=1
            )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        plt.tight_layout(pad=0)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(image)
        plt.close(fig)
        print(f"Frame {f} done")
    imageio.mimsave(output_gif, frames, fps=1/delay, loop=0)
    print(f"Top-view Gaussian heatmap animation saved as {output_gif}")

def main():
    parser = argparse.ArgumentParser(description="Full pipeline: video -> tracking -> topview+gaussian GIF, with time in trajectories_packets")
    parser.add_argument('--video', type=str, required=True, help='Input video file')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints_pipeline', help='Directory for checkpoints')
    parser.add_argument('--traj_file', type=str, default='trajectories_pipeline.parquet', help='Trajectory output file')
    parser.add_argument('--trajectories_packets', type=str, default='trajectories_packets.parquet', help='Trajectory file with time')
    parser.add_argument('--output_gif', type=str, default='gaussian_topview.gif', help='Output GIF filename')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay (seconds) between frames')
    parser.add_argument('--sigma', type=float, default=10, help='Gaussian sigma for heatmap')
    parser.add_argument('--interval', type=float, default=0.2, help='Interval for detection/tracking')
    args = parser.parse_args()

    # Step 1: Run detection/tracking
    traj_file_found = run_detection_tracking(args.video, args.checkpoints_dir, args.traj_file, interval=args.interval)

    # Step 2: Add time column and save as trajectories_packets
    packets_file = add_time_to_trajectories(traj_file_found, args.interval, args.trajectories_packets)

    # Step 3: Generate topview+gaussian GIF
    run_gaussian_topview(packets_file, output_gif=args.output_gif, delay=args.delay, sigma=args.sigma)

if __name__ == '__main__':
    main()