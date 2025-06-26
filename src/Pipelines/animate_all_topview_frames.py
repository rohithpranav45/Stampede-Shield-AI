import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
import argparse
from utils import get_grid_edges, step_to_seconds
import yaml

def get_panic_trigger_step(config_path='pipeline_config.yaml'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg['panic_simulation']['panic_trigger_step']

def get_first_stampede_step(alerts_path='panic_monitor_frames/pressure_alerts.csv'):
    df = pd.read_csv(alerts_path)
    if 'start_step' in df.columns:
        return int(df['start_step'].min())
    elif len(df) > 0:
        return int(df.iloc[:,2].min())  # fallback: 3rd column is start_step
    return None

def animate_all_topview_frames(
    traj_csv, output_gif='all_topview_frames.gif', delay=0.2, pause_frames=5
):
    panic_trigger_step = get_panic_trigger_step()
    first_stampede_step = get_first_stampede_step()
    important_steps = [panic_trigger_step]
    if first_stampede_step is not None:
        important_steps.append(first_stampede_step)

    df = pd.read_csv(traj_csv)
    steps = df['step'].max() + 1
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    x_edges, y_edges = get_grid_edges(x_min, x_max, y_min, y_max, 40)
    frames = []
    for t in range(steps):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('black')
        step_df = df[df['step'] == t]
        for _, row in step_df.iterrows():
            color = 'red' if row['panic'] else 'gray'
            ax.scatter(row['x'], row['y'], color=color, s=40, edgecolor='white', zorder=10)
        # Highlight important frames
        if t == panic_trigger_step:
            ax.set_title(f'Step {t} (t={step_to_seconds(t):.1f}s) - PANIC TRIGGER', color='pink')
            for spine in ax.spines.values():
                spine.set_edgecolor('pink')
                spine.set_linewidth(4)
            ax.text(0.5, 1.05, 'PANIC TRIGGER', color='pink', fontsize=16, ha='center', va='bottom', transform=ax.transAxes)
        elif t == first_stampede_step:
            ax.set_title(f'Step {t} (t={step_to_seconds(t):.1f}s) - STAMPEDE', color='magenta')
            for spine in ax.spines.values():
                spine.set_edgecolor('magenta')
                spine.set_linewidth(4)
            ax.text(0.5, 1.05, 'FIRST STAMPEDE', color='magenta', fontsize=16, ha='center', va='bottom', transform=ax.transAxes)
        else:
            ax.set_title(f'Step {t} (t={step_to_seconds(t):.1f}s)')
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_xlabel('x (meters)')
        ax.set_ylabel('y (meters)')
        plt.tight_layout(pad=0)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        # Pause on important frames
        if t in important_steps:
            frames.extend([image] * pause_frames)
        else:
            frames.append(image)
        plt.close(fig)
        print(f"All Topview Frame {t} done")
    imageio.mimsave(output_gif, frames, fps=1/delay, loop=0)
    print(f"All top-view animation saved as {output_gif}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Animate all SFM simulation frames in top-view point cloud style.')
    parser.add_argument('--traj_csv', type=str, default='panic_sfm_trajectories.csv', help='CSV of simulated SFM trajectories')
    parser.add_argument('--output_gif', type=str, default='all_topview_frames.gif', help='Output GIF filename')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay (seconds) between frames')
    parser.add_argument('--pause_frames', type=int, default=5, help='Number of frames to pause on important steps')
    args = parser.parse_args()
    animate_all_topview_frames(args.traj_csv, output_gif=args.output_gif, delay=args.delay, pause_frames=args.pause_frames)