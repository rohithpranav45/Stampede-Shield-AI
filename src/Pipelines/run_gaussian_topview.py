import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage import gaussian_filter
import argparse

def get_color(id):
    np.random.seed(int(id))
    return np.random.rand(3,)

def run_gaussian_topview(traj_file, output_gif="gaussian_topview.gif", delay=0.2, sigma=10):
    # Load trajectory data
    if traj_file.endswith('.parquet'):
        df = pd.read_parquet(traj_file)
    else:
        df = pd.read_csv(traj_file)
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
            # Create a 2D histogram for the points
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run top-view 2D Gaussian heatmap with points and locus lines.")
    parser.add_argument('--traj_file', type=str, required=True, help='Trajectory Parquet or CSV file (from DeepSORT logging)')
    parser.add_argument('--output_gif', type=str, default='gaussian_topview.gif', help='Output GIF filename')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay (seconds) between frames')
    parser.add_argument('--sigma', type=float, default=10, help='Gaussian sigma for heatmap')
    args = parser.parse_args()
    run_gaussian_topview(args.traj_file, output_gif=args.output_gif, delay=args.delay, sigma=args.sigma) 