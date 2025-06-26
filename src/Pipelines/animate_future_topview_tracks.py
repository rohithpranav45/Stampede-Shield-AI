import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage import gaussian_filter
import argparse

def sph_kernel(r, h):
    return np.exp(-0.5 * (r / h) ** 2)

def forecast_future_positions(df, history_window=5, forecast_steps=10, dt=1, h=5):
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
    # SPH-like forecast for all steps
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
    return ids, all_pred_positions

def animate_future_topview_tracks(traj_file, output_gif="future_topview_tracks.gif", history_window=5, forecast_steps=10, dt=1, h=5, delay=0.2):
    # Load trajectory data
    if traj_file.endswith('.parquet'):
        df = pd.read_parquet(traj_file)
    else:
        df = pd.read_csv(traj_file)
    ids, all_pred_positions = forecast_future_positions(df, history_window, forecast_steps, dt, h)
    # Get plot limits from original data
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    id_colors = {id_: np.random.rand(3,) for id_ in ids}
    frames = []
    for t, positions in enumerate(all_pred_positions):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('black')
        # Draw predicted tracks up to this frame
        for i, id_ in enumerate(ids):
            color = id_colors[id_]
            # Draw path up to current forecast step
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
        frames.append(image)
        plt.close(fig)
        print(f"Future Frame {t} done")
    imageio.mimsave(output_gif, frames, fps=1/delay, loop=0)
    print(f"Future top-view tracking animation saved as {output_gif}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate future 2D top-view with SPH-forecasted tracking lines for each ID.")
    parser.add_argument('--traj_file', type=str, required=True, help='Trajectory Parquet or CSV file (from DeepSORT logging)')
    parser.add_argument('--output_gif', type=str, default='future_topview_tracks.gif', help='Output GIF filename')
    parser.add_argument('--history_window', type=int, default=5, help='Number of frames to use for velocity estimation')
    parser.add_argument('--forecast_steps', type=int, default=10, help='Number of forecast steps')
    parser.add_argument('--dt', type=float, default=1, help='Time step for forecast')
    parser.add_argument('--h', type=float, default=5, help='SPH kernel bandwidth')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay (seconds) between frames')
    args = parser.parse_args()
    animate_future_topview_tracks(args.traj_file, output_gif=args.output_gif, history_window=args.history_window, forecast_steps=args.forecast_steps, dt=args.dt, h=args.h, delay=args.delay) 