import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage import gaussian_filter
import argparse

# --- Helper: SPH kernel and forecast logic (from animate_future_topview_tracks.py) ---
def sph_kernel(r, h):
    return np.exp(-0.5 * (r / h) ** 2)

def forecast_future_positions(df, history_window=5, forecast_steps=10, dt=1, h=5):
    print("[INFO] Forecasting future positions...")
    df = df.sort_values(['id', 'frame'])
    last_frames = df.groupby('id').tail(history_window)
    velocities = last_frames.groupby('id').apply(
        lambda g: pd.Series({
            'x': g['x'].iloc[-1],
            'y': g['y'].iloc[-1],
            'vx': np.mean(g['x'].diff().iloc[1:] / g['frame'].diff().iloc[1:]),
            'vy': np.mean(g['y'].diff().iloc[1:] / g['frame'].diff().iloc[1:])
        })
    ).reset_index()
    ids = velocities['id'].values
    positions = velocities[['x', 'y']].values
    velocities_arr = velocities[['vx', 'vy']].values
    all_pred_positions = [positions.copy()]
    pred_positions = positions.copy()
    pred_velocities = velocities_arr.copy()
    for step in range(forecast_steps):
        new_positions = pred_positions.copy()
        new_velocities = pred_velocities.copy()
        for i, pos in enumerate(pred_positions):
            r = np.linalg.norm(pred_positions - pos, axis=1)
            w = sph_kernel(r, h)
            v_avg = np.average(pred_velocities, axis=0, weights=w)
            new_velocities[i] = v_avg
            new_positions[i] = pos + v_avg * dt
        pred_positions = new_positions
        pred_velocities = new_velocities
        all_pred_positions.append(pred_positions.copy())
        print(f"[INFO] Forecast step {step+1}/{forecast_steps} complete.")
    return ids, all_pred_positions

# --- Main animation logic ---
def draw_topview_frame(df, ids, id_colors, frame_num, x_min, x_max, y_min, y_max):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor('black')
    for id_ in ids:
        track = df[(df['id'] == id_) & (df['frame'] <= frame_num)]
        if len(track) == 0:
            continue
        color = id_colors[id_]
        ax.plot(track['x'], track['y'], color=color, linewidth=2)
        ax.scatter(track['x'].iloc[-1], track['y'].iloc[-1], color=color, s=40, edgecolor='white', zorder=10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return image

def draw_forecast_topview_frame(all_pred_positions, ids, id_colors, t, x_min, x_max, y_min, y_max):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor('black')
    for i, id_ in enumerate(ids):
        color = id_colors[id_]
        path = np.array([all_pred_positions[s][i] for s in range(t+1)])
        ax.plot(path[:,0], path[:,1], color=color, linewidth=2)
        ax.scatter(path[-1,0], path[-1,1], color=color, s=40, edgecolor='white', zorder=10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return image

def animate_side_by_side_with_forecast(video_file, traj_file, output_gif="side_by_side_forecast.gif", history_window=5, forecast_steps=10, dt=1, h=5, delay=0.2):
    print(f"[INFO] Loading trajectory data from {traj_file} ...")
    if traj_file.endswith('.parquet'):
        df = pd.read_parquet(traj_file)
    else:
        df = pd.read_csv(traj_file)
    ids = df['id'].unique()
    id_colors = {id_: np.random.rand(3,) for id_ in ids}
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    print(f"[INFO] Loaded {len(df)} trajectory points for {len(ids)} IDs.")
    # Forecast future positions
    forecast_ids, all_pred_positions = forecast_future_positions(df, history_window, forecast_steps, dt, h)
    print(f"[INFO] Opening video file {video_file} ...")
    cap = cv2.VideoCapture(video_file)
    video_frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frames.append(frame_rgb)
        if frame_idx % 10 == 0:
            print(f"[INFO] Loaded video frame {frame_idx}")
        frame_idx += 1
    cap.release()
    n_video_frames = len(video_frames)
    print(f"[INFO] Total video frames loaded: {n_video_frames}")
    # Prepare topview frames for each video frame
    frame_nums = sorted(df['frame'].unique())
    # Map video frame index to closest available frame in trajectory
    frame_map = []
    for i in range(n_video_frames):
        if i in frame_nums:
            frame_map.append(i)
        else:
            closest = min(frame_nums, key=lambda x: abs(x-i))
            frame_map.append(closest)
    frames = []
    print("[INFO] Generating synchronized video+topview frames...")
    for i in range(n_video_frames):
        video_img = video_frames[i]
        topview_img = draw_topview_frame(df, ids, id_colors, frame_map[i], x_min, x_max, y_min, y_max)
        h1, w1, _ = video_img.shape
        h2, w2, _ = topview_img.shape
        target_h = max(h1, h2)
        video_img_resized = cv2.resize(video_img, (int(w1 * target_h / h1), target_h))
        topview_img_resized = cv2.resize(topview_img, (int(w2 * target_h / h2), target_h))
        combined = np.concatenate([video_img_resized, topview_img_resized], axis=1)
        frames.append(combined)
        if i % 10 == 0:
            print(f"[INFO] Synchronized frame {i+1}/{n_video_frames}")
    print("[INFO] Generating forecast-only frames...")
    blank_left = np.zeros_like(video_img_resized)
    for t in range(1, len(all_pred_positions)):
        forecast_img = draw_forecast_topview_frame(all_pred_positions, forecast_ids, id_colors, t, x_min, x_max, y_min, y_max)
        forecast_img_resized = cv2.resize(forecast_img, (topview_img_resized.shape[1], topview_img_resized.shape[0]))
        combined = np.concatenate([blank_left, forecast_img_resized], axis=1)
        frames.append(combined)
        if t % 2 == 0:
            print(f"[INFO] Forecast frame {t}/{len(all_pred_positions)-1}")
    print(f"[INFO] Saving output GIF to {output_gif} ...")
    imageio.mimsave(output_gif, frames, fps=1/delay, loop=0)
    print(f"[SUCCESS] Side-by-side animation with forecast saved as {output_gif}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate side-by-side: left=video, right=topview, then forecast only.")
    parser.add_argument('--video_file', type=str, required=True, help='Input video file')
    parser.add_argument('--traj_file', type=str, required=True, help='Trajectory Parquet or CSV file')
    parser.add_argument('--output_gif', type=str, default='side_by_side_forecast.gif', help='Output GIF filename')
    parser.add_argument('--history_window', type=int, default=5, help='Number of frames to use for velocity estimation')
    parser.add_argument('--forecast_steps', type=int, default=10, help='Number of forecast steps')
    parser.add_argument('--dt', type=float, default=1, help='Time step for forecast')
    parser.add_argument('--h', type=float, default=5, help='SPH kernel bandwidth')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay (seconds) between frames')
    args = parser.parse_args()
    animate_side_by_side_with_forecast(
        args.video_file, args.traj_file, output_gif=args.output_gif,
        history_window=args.history_window, forecast_steps=args.forecast_steps,
        dt=args.dt, h=args.h, delay=args.delay
    ) 