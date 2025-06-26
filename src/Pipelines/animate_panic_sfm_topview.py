import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
import argparse
from utils import get_grid_edges, step_to_seconds

def animate_panic_sfm_topview(traj_csv, output_gif='panic_sfm_topview.gif', delay=0.2):
    df = pd.read_csv(traj_csv)
    ids = df['id'].unique()
    steps = df['step'].max() + 1
    # Assign a color to each agent
    id_colors = {id_: np.random.rand(3,) for id_ in ids}
    # Get plot limits
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    x_edges, y_edges = get_grid_edges(x_min, x_max, y_min, y_max, 40)  # grid_size is arbitrary for axis, adjust as needed
    frames = []
    for t in range(steps):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('black')
        for id_ in ids:
            agent_traj = df[(df['id'] == id_) & (df['step'] <= t)]
            if len(agent_traj) == 0:
                continue
            color = id_colors[id_]
            # Draw path
            ax.plot(agent_traj['x'], agent_traj['y'], color=color, linewidth=2)
            # Draw current position, color by panic state
            last = agent_traj.iloc[-1]
            panic = last['panic']
            marker_color = 'red' if panic else color
            ax.scatter(last['x'], last['y'], color=marker_color, s=40, edgecolor='white', zorder=10)
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_xlabel('x (meters)')
        ax.set_ylabel('y (meters)')
        ax.set_title(f'Step {t} (t={step_to_seconds(t):.1f}s)')
        plt.tight_layout(pad=0)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(image)
        plt.close(fig)
        print(f"Panic SFM Frame {t} done")
    imageio.mimsave(output_gif, frames, fps=1/delay, loop=0)
    print(f"Panic SFM top-view animation saved as {output_gif}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Animate SFM panic simulation in top-view format.')
    parser.add_argument('--traj_csv', type=str, default='panic_sfm_trajectories.csv', help='CSV of simulated SFM trajectories')
    parser.add_argument('--output_gif', type=str, default='panic_sfm_topview.gif', help='Output GIF filename')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay (seconds) between frames')
    args = parser.parse_args()
    animate_panic_sfm_topview(args.traj_csv, output_gif=args.output_gif, delay=args.delay) 